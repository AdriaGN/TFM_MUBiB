"""Mòdul encarregat de realitzar l'entrenament de la xarxa neuronal."""

import logging
import sys
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from carregador_dades import obtenir_dataloaders
from configuracio import parametres
from matplotlib import pyplot as plt
from model import XarxaNeuronal
from monai.losses import SSIMLoss
from torch.utils.tensorboard import SummaryWriter


def entrenament_model() -> None:
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
    """
    ### 1. Preparació de l'entorn d'entrenament
    # Inicialització del sistema de logging, dispositiu i registre de la configuració
    _configurar_logging()
    log_tensorboard = SummaryWriter(parametres.DIR_TENSORBOARD)
    dispositiu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _registrar_configuracio_model(dispositiu)

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
    criteri_ssim = SSIMLoss(spatial_dims=3)

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
    loader_entrenament, loader_validacio = obtenir_dataloaders()
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
            inputs = batch["imatge"].to(dispositiu)
            optimitzador.zero_grad()

            # Utilitzar l'accelerador AMP per calcular els outputs i calcular pèrdues
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                perdua_l1 = criteri_l1(outputs, inputs)
                perdua_ssim = criteri_ssim(outputs, inputs)
                perdua_total = (parametres.PES_L1 * perdua_l1) + (
                    parametres.PES_SSIM * perdua_ssim
                )

            # Retropropagació i actualització dels errors amb escalat per evitar pèrdues
            # si són molt petits.
            escalador.scale(perdua_total).backward()
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
                inputs = batch["imatge"].to(dispositiu)
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    perdua_l1 = criteri_l1(outputs, inputs)
                    perdua_ssim = criteri_ssim(outputs, inputs)
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

        # Log a la TensorBoard de cada pèrdua i la taxa d'aprenentatge com a escalars
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


def _configurar_logging() -> None:
    """Configuració del sistema de logs de l'entrenament tant a consola com a un fitxer.

    Els logs s'emmagatzemaran tant en la consola d'execució com en un fitxer alhora. Hi
    haurà diferents nivells de missatges (INFO, ERROR, etc.) i l'estructura del format
    serà HORES:MINUTS:SEGONS - NIVELL MISSATGE - MISSATGE
    """
    # Creació del format del log
    format_log = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Assegurar que les carpetes de logs i evolució existeixen
    parametres.DIR_LOGS.mkdir(parents=True, exist_ok=True)
    parametres.DIR_EVOLUCIO.mkdir(parents=True, exist_ok=True)

    # Creació dels dos handlers (consola i fitxer) per gestionar els logs
    handler_consola = logging.StreamHandler(sys.stdout)
    handler_consola.setFormatter(format_log)
    handler_logs = logging.FileHandler(parametres.RUTA_FITXER_LOGS, encoding="UTF-8")
    handler_logs.setFormatter(format_log)

    # Configuració del logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler_consola)
    logger.addHandler(handler_logs)


def _registrar_configuracio_model(dispositiu: torch.device) -> None:
    """Escriu una entrada en el fitxer de log amb la configuració del model entrenat.

    Llegeix la dataclass de la configuració (emmagatzemada a parametres) i genera un
    diccionari dels seus valors amb asdict. Tot seguit, per cada parella clau-valor,
    escriu una línia en el log, formatejant de manera que hi hagi 25 caràcters a la
    part de clau. A més, al final també inclou el dispositiu que s'ha utilitzat.
    S'encapsula tot amb "=" per diferenciar-ho de la resta de logs.

    Args:
        dispositiu: Dispositiu (GPU o CPU) amb el que s'entrena el model.
    """
    # Capçalera
    logging.info("=" * 50)
    logging.info(f"CONFIGURACIÓ MODEL {parametres.NOM_MODEL}:\n")
    # Configuració i dispositiu
    diccionari_configuracio = asdict(parametres)
    for clau, valor in diccionari_configuracio.items():
        logging.info(f"{clau:<25}: {valor}")
    logging.info(f"\nDispositiu utilitzat durant l'entrenament: {dispositiu}")
    # Tancament
    logging.info("=" * 50 + "\n")


def _generar_foto_evolucio(
    model: torch.nn.Module, imatge: torch.Tensor, epoca: int
) -> None:
    """Genera una comparativa visual entre la imatge d'input i output en el mateix punt.

    Es generarà una imatge on es compararà la imatge original d'un tall central del
    cervell amb la generada per el model en el mateix punt.

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

    # Extracció d'un tall central (96) i convertir de tensor a matriu de Numpy
    # per poder fer les imatges 2D
    original = imatge[0, 0, :, :, 96].cpu().numpy()
    prediccio = prediccio[0, 0, :, :, 96].cpu().numpy()

    # Creació de la imatge
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Evolució del model {parametres.NOM_MODEL} a l'època {epoca + 1}")
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(prediccio, cmap="gray")
    axs[1].set_title("Predicció")
    axs[1].axis("off")

    # Guardar la imatge en el directori indicat a la configuració i tancament
    ruta_guardat = parametres.DIR_EVOLUCIO / f"evolucio_epoca_{epoca + 1}.png"
    plt.savefig(ruta_guardat, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"Captura de l'evolució d'entrenament guardada: {ruta_guardat.name}")


if __name__ == "__main__":
    """Execució principal de l'entrenament."""
    # Gestionar errors durant l'execució i loggejar-los
    try:
        entrenament_model()
    except KeyboardInterrupt:
        logging.warning("Entrenament interromput manualment")
    except Exception as e:
        logging.error(
            f"Entrenament interromput per un error. Detalls:\n{e}, exc_info=True"
        )
