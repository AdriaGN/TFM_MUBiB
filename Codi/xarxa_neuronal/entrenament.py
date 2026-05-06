"""Mòdul encarregat de realitzar l'entrenament de la xarxa neuronal."""

import logging

import matplotlib

matplotlib.use("Agg")  # Evitar que Matplotlib mostri imatges durant el procés
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from configuracio import parametres
from dades.carregador_dades import obtenir_dataloaders_entrenament
from matplotlib import pyplot as plt
from monai.losses import SSIMLoss
from torch.utils.tensorboard import SummaryWriter

from xarxa_neuronal.model import XarxaNeuronal


def entrenament_model(dispositiu: torch.device) -> bool:
    """Funció principal que dirigeix l'entrenament de la xarxa neuronal.

    Es comença amb la preparació de l'entorn d'entrenament inicialitzant els sistemes de
    logging, configuració, etc, i es configuren els components principals necessàris
    durant l'entrenament. A més, també es farà un banc de proves amb les diferents
    configuracions de PyTorch per augmentar molt el rendiment. Tot seguit, es selecciona
    una imatge per fer la comparació visual (sempre la mateixa) i es busca si hi ha un
    checkpoint d'execucions prèvies. Llavors es comença l'entrenament, utilitzant eines
    com AMP de PyTorch (per millorar el rendiment de la GPU i que necessiti menys
    memòria), i s'entrena per batches (on es guardarà els valors de les pèrdues de cada
    iteració). Tot seguit es fa la validació del mateix batch, es calculen els resultats
    finals i s'escriuen logs tant a un fitxer com a la TensorBoard per tenir
    traçabilitat. Finalment es verifica si el model generat és millor que anteriors i es
    guarda per evitar sobreajustaments.

    Args:
        dispositiu: Dispositiu (GPU o CPU) amb el que s'entrena el model.

    Returns:
        Booleà que indica si s'ha entrenat correctament o no el model.
    """
    ### 1. Preparació de l'entorn d'entrenament
    # Inicialització del sistema de logging de Tensorboad amb les mètriques
    log_tensorboard = SummaryWriter(parametres.DIR_TENSORBOARD)

    # Benchmark automàtic per trobar la millor configuració d'entrenament per a cada
    # capa del model i millorar el rendiment
    torch.backends.cudnn.benchmark = True

    # Inicialització dels components principals de l'entrenament (model, optimitzador,
    # reductor de la taxa d'aprenentatge, criteris de la pèrdua, i escalador AMP)
    model = XarxaNeuronal().to(dispositiu)

    optimitzador = optim.AdamW(
        model.parameters(),
        lr=parametres.TAXA_APRENENTATGE,
        weight_decay=parametres.CAIGUDA_PES_ADAMW,
    )

    planificador_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimitzador,
        mode="min",
        patience=parametres.ITERACIONS_REDUCCIO,
        factor=parametres.FACTOR_REDUCCIO,
    )

    criteri_l1 = nn.L1Loss()
    escalador = torch.amp.GradScaler("cuda")

    # Verificar si existeix algun checkpoint i carregar-lo
    perdua_minima_validacio = float("inf")  # Assegurar la selecció de qualsevol valor
    epoca_inicial = 0

    if parametres.RUTA_PESOS_ULTIM.exists():
        checkpoint = torch.load(parametres.RUTA_PESOS_ULTIM, weights_only=False)
        model.load_state_dict(checkpoint["estat_model"])
        optimitzador.load_state_dict(checkpoint["estat_optimitzador"])
        planificador_lr.load_state_dict(checkpoint["estat_planificador"])
        escalador.load_state_dict(checkpoint["estat_escalador"])
        epoca_inicial = checkpoint["epoca"] + 1
        perdua_minima_validacio = checkpoint["perdua_minima_validacio"]
        logging.info(
            f"Checkpoint carregat. Continuant entrenament des de l'època "
            f"{epoca_inicial + 1}"
        )

    ### 2. Entrenament del model
    logging.info("=== Inici de l'entrenament ===")

    # Càrrega de les dades amb els DataLoaders (utilitzant type: ignore per ignorar el
    # nombre de caràcters màxims per línia en el log)
    loader_entrenament, loader_validacio = obtenir_dataloaders_entrenament()
    logging.info(
        f"Carregades {len(loader_entrenament.dataset)} mostres d'entrenament i "  # type: ignore
        f"{len(loader_validacio.dataset)} de validació."  # type: ignore
    )

    # Seleccionar la imatge de mostra per comparar l'evolució gràfica una sola vegada
    imatge_mostra = next(iter(loader_validacio))["imatge"][0:1].to(dispositiu)

    # Bucle principal d'entrenament per èpoques
    for epoca in range(epoca_inicial, parametres.EPOQUES):
        logging.info(f"--- Època {epoca + 1} / {parametres.EPOQUES} ---")

        ## 2.1 Fase d'entrenament
        # Carregar model i inicialitzar a 0 les pèrdues d'entrenament, L1 i SSIm
        model.train()
        suma_perdua_entrenament = 0.0
        suma_l1_entrenament = 0.0
        suma_ssim_entrenament = 0.0

        # Bucle d'entrenament per batches
        for i, batch in enumerate(loader_entrenament):
            # Carregar imatges a la GPU i reiniciar les gradients de l'optimitzador
            optimitzador.zero_grad()

            # Utilitzar l'accelerador AMP per calcular els outputs i calcular pèrdues
            with torch.amp.autocast("cuda"):
                inputs_nets = batch["imatge"].to(dispositiu)
                # Aplicar soroll a la imatge per reduir la probabilitat de que la xarxa
                # neuronal faci còpies de les imatges en comptes d'aprendre
                soroll = torch.randn_like(inputs_nets) * parametres.FACTOR_SOROLL
                inputs_bruts = inputs_nets + soroll
                outputs = model(inputs_bruts)
                # Pel cas de la pèrdua de L1 es pot utilitzar el criteri anterior, però
                # per SSIM cal adaptar-se al rang de colors del batch (degut a la
                # normalització realitzada per MONAI)
                perdua_l1 = criteri_l1(outputs, inputs_nets)
                rang_batch = (inputs_nets.max() - inputs_nets.min()).item()
                criteri_ssim_batch = SSIMLoss(spatial_dims=3, data_range=rang_batch)
                perdua_ssim = criteri_ssim_batch(outputs, inputs_nets)
                # Càlcul de la pèrdua total a partir dels pesos configurats
                perdua_total = (parametres.PES_L1 * perdua_l1) + (
                    parametres.PES_SSIM * perdua_ssim
                )

            # Retropropagació i actualització dels errors amb escalat per evitar pèrdues
            # si són molt petits. També fa un tall de gradients seguint la configuració,
            # per evitar que el model es trenqui amb datasets petits
            escalador.scale(perdua_total).backward()
            if parametres.TALL_GRADIENTS:
                escalador.unscale_(optimitzador)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=parametres.PES_GRADIENTS
                )
            escalador.step(optimitzador)
            escalador.update()

            # Suma de la pèrdua de l'entrenament total, L1 i SSIM per fer-ne la mitjana
            suma_perdua_entrenament += perdua_total.item()
            suma_l1_entrenament += perdua_l1.item()
            suma_ssim_entrenament += perdua_ssim.item()

            # Mostrar per pantalla els resultats de cada iteracio
            logging.info(
                f"Batch {i + 1}/{len(loader_entrenament)} - "
                f"Pèrdua total: {perdua_total.item():.4f} - "
                f"Pèrdua L1: {perdua_l1.item():.4f} - "
                f"Pèrdua SSIM: {perdua_ssim.item():.4f}",
            )

        ## 2.2 Fase de validació
        # Carregar model en avaluació i posar a 0 les pèrdues de validació, L1 i SSIm
        model.eval()
        suma_perdua_validacio = 0.0
        suma_l1_validacio = 0.0
        suma_ssim_validacio = 0.0

        # Utilitzar torch.nograd() per indicar que es fa una validació i indicar a
        # PyTorch que no calculi els gradients i només validi.
        with torch.no_grad():
            # Bucle de validació per batches
            for batch in loader_validacio:
                # Carregar imatges a la GPU i utilitzar l'accelerador AMP de nou
                inputs_nets = batch["imatge"].to(dispositiu)
                # Com abans, aplicar soroll a la imatge per reduir la probabilitat de
                # que la xarxa neuronal faci còpies de les imatges en comptes d'aprendre
                soroll = torch.randn_like(inputs_nets) * parametres.FACTOR_SOROLL
                inputs_bruts = inputs_nets + soroll
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs_bruts)
                    # Com abans, calcular la pèrdua de L1 amb el seu criteri i la de
                    # SSIM per rangs
                    perdua_l1 = criteri_l1(outputs, inputs_nets)
                    rang_batch = (inputs_nets.max() - inputs_nets.min()).item()
                    criteri_ssim_batch = SSIMLoss(spatial_dims=3, data_range=rang_batch)
                    perdua_ssim = criteri_ssim_batch(outputs, inputs_nets)
                    # Càlcul de la pèrdua total
                    perdua_total = (parametres.PES_L1 * perdua_l1) + (
                        parametres.PES_SSIM * perdua_ssim
                    )

                # Sumar pèrdues de validació total, L1 i SSIM per fer mitjanes
                suma_perdua_validacio += perdua_total.item()
                suma_l1_validacio += perdua_l1.item()
                suma_ssim_validacio += perdua_ssim.item()

        ## 2.3 Càlcul de mitjanes de l'època
        mitjana_entrenament = suma_perdua_entrenament / len(loader_entrenament)
        mitjana_l1_entrenament = suma_l1_entrenament / len(loader_entrenament)
        mitjana_ssim_entrenament = suma_ssim_entrenament / len(loader_entrenament)
        mitjana_validacio = suma_perdua_validacio / len(loader_validacio)
        mitjana_l1_validacio = suma_l1_validacio / len(loader_validacio)
        mitjana_ssim_validacio = suma_ssim_validacio / len(loader_validacio)

        # Revisar si l'aprenentatge de validació ha millorat per si cal que el
        # planificador redueixi a la meitat la taxa d'aprenentatge i actualitzar-la
        planificador_lr.step(mitjana_validacio)
        lr_actual = optimitzador.param_groups[0]["lr"]

        ### 3. Log dels resultats i imatges de la època
        # Log al fitxer de logs
        logging.info(
            f"Pèrdua Entrenament: Total {mitjana_entrenament:.6f} - "
            f"L1: {mitjana_l1_entrenament:.4f} - "
            f"SSIM: {mitjana_ssim_entrenament:.4f}"
        )
        logging.info(
            f"Pèrdua Validació: Total {mitjana_validacio:.6f} - "
            f"L1: {mitjana_l1_validacio:.4f} - "
            f"SSIM: {mitjana_ssim_validacio:.4f}"
        )

        # Logs a la TensorBoard de cada pèrdua i la taxa d'aprenentatge com a escalars
        log_tensorboard.add_scalars(
            "Pèrdua Total",
            {"Entrenament": mitjana_entrenament, "Validacio": mitjana_validacio},
            epoca,
        )
        log_tensorboard.add_scalars(
            "Pèrdua L1",
            {"Entrenament": mitjana_l1_entrenament, "Validacio": mitjana_l1_validacio},
            epoca,
        )
        log_tensorboard.add_scalars(
            "Pèrdua SSIM",
            {
                "Entrenament": mitjana_ssim_entrenament,
                "Validacio": mitjana_ssim_validacio,
            },
            epoca,
        )
        log_tensorboard.add_scalar("Taxa_Aprenentatge", lr_actual, epoca)

        # Creació de la imatge evolutiva de la època durant les 5 primeres èpoques i
        # cada època posterior múltiple de 5
        if epoca < 5 or (epoca + 1) % 5 == 0:
            _generar_foto_evolucio(model, imatge_mostra, epoca)

        ### Tancament de l'entrenament de l'època i creació del checkpoint
        # Crear i guardar els resultats de la darrera època
        checkpoint_actual = {
            "estat_model": model.state_dict(),
            "estat_optimitzador": optimitzador.state_dict(),
            "estat_planificador": planificador_lr.state_dict(),
            "estat_escalador": escalador.state_dict(),
            "epoca": epoca,
            "perdua_minima_validacio": perdua_minima_validacio,
        }
        torch.save(checkpoint_actual, parametres.RUTA_PESOS_ULTIM)

        # Verificar si els resultats són millors que en èpoques anteriors i
        # sobreescriure millor època en tal cas per evitar sobreajustament
        if mitjana_validacio < perdua_minima_validacio:
            perdua_minima_validacio = mitjana_validacio
            torch.save(model.state_dict(), parametres.RUTA_PESOS_MILLOR)
            logging.info(
                f"Nou millor model ({epoca + 1}) guardat amb pèrdua de validació de "
                f"{perdua_minima_validacio:.6f}"
            )

    log_tensorboard.close()
    logging.info("=== Final de l'entrenament ===")
    return True


def _generar_foto_evolucio(
    model: torch.nn.Module, imatge: torch.Tensor, epoca: int
) -> None:
    """Genera una comparativa visual entre la imatge d'input i output en el mateix punt.

    Es generarà una imatge on es compararà la imatge original d'un tall central del
    cervell amb la generada per el model en el mateix punt. Pot donar-se la situació
    en que la imatge sigui d'una època amb pitjors resultats, però es logejarà igualment
    per traçabilitat i històric.

    Args:
        model: Model entrenat en la època actual en que s'ha cridat la funció.
        imatge: Tensor amb la imatge de mostra triada abans de començar l'entrenament.
        epoca: Nombre de l'època actual.
    """
    # Carregar el model en mode avaluació i reconstruïr la imatge amb torch.no_grad()
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            prediccio = model(imatge)

    # Extracció d'un tall central (96) i convertir de tensor a matriu de Numpy per poder
    # fer les imatges 2D i treballar amb MatPlotlib
    original = imatge[0, 0, :, :, 96].cpu().float().numpy()
    prediccio = prediccio[0, 0, :, :, 96].cpu().float().numpy()

    # Comprovació dels valors de colors mínim i màxim per representar-la correctament
    # (en cas contrari, la imatge tindria un color diferent durant la captura i no es
    # podria comparar correctament)
    v_min, v_max = np.percentile(original, [1, 99])

    # Creació de la imatge
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Evolució del model {parametres.NOM_MODEL} a l'època {epoca + 1}")
    axs[0].imshow(original, cmap="gray", vmin=v_min, vmax=v_max)
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(prediccio, cmap="gray", vmin=v_min, vmax=v_max)
    axs[1].set_title("Predicció")
    axs[1].axis("off")

    # Guardar la imatge en el directori indicat a la configuració i tancament
    ruta_guardat = parametres.DIR_EVOLUCIO / f"evolucio_epoca_{epoca + 1}.png"
    plt.savefig(ruta_guardat, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"Captura de l'evolució d'entrenament guardada: {ruta_guardat.name}")
