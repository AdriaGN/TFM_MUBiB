"""Orquestrador principal per a l'execució de tots els processos del projecte."""

import logging
import sys
from dataclasses import asdict
from typing import Any

import torch
from configuracio import parametres
from dades.preparacio_dataset import divisio_dades, extraccio_cervells
from inferencia.calculador_dice import calcular_dice_brats
from inferencia.calibrador_models import calibrar_model
from inferencia.generador_estadistiques import generar_estadistiques_i_rendiment
from inferencia.generador_grafiques import generar_grafics
from inferencia.generador_metriques import generacio_metriques
from inferencia.harmonitzador_metriques import aplicar_neurocombat
from xarxa_neuronal.entrenament import entrenament_model


def executar_pipeline() -> None:
    """'Menú' principal que executa la gestió de tot el projecte.

    Genera primerament un diccionari amb els models i els seus datasets d'entrenament i
    demana a l'usuari quines opcions vol fer (actualment pot triar entre fer la neteja
    de cervells amb Skull-Stripping, fer l'entrenament o fer l'anàlisi d'inferència).
    """
    # Conjunt de models a entrenar amb els seus datasets
    models_datasets = [
        # Models individuals
        {"nom": "Model_HCP", "datasets": ("HCP",)},
        {"nom": "Model_IXI", "datasets": ("IXI",)},
        {"nom": "Model_OASIS", "datasets": ("OASIS",)},
        # Models combinatoris
        {"nom": "Model_IXI_HCP", "datasets": ("IXI", "HCP")},
        {"nom": "Model_IXI_OASIS", "datasets": ("IXI", "OASIS")},
        {"nom": "Model_HCP_OASIS", "datasets": ("HCP", "OASIS")},
        # Model global
        {"nom": "Model_IXI_HCP_OASIS", "datasets": ("IXI", "HCP", "OASIS")},
    ]

    # Obtenir informació del dispositiu amb que es farà els processos
    dispositiu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mostrar les opcions i esperar a la tria de l'usuari
    print("1. Skull Stripping dels cervells")
    print("2. Entrenament i calibració dels models")
    print("3. Anàlisi d'inferència")
    opcio = input()

    # Switch-case amb les diferents opcions i retorn del resultat a l'usuari
    match opcio:
        case "1":
            try:
                extraccio_cervells()
            except Exception as error:
                print(f"Error durant l'Skull-Stripping: {error}")
        case "2":
            try:
                print("Iniciant entrenament dels models")
                retorn_entrenament = _orquestrar_entrenament(
                    models_datasets, dispositiu
                )
                # Notificar resultat de l'execució
                if retorn_entrenament:
                    print("Entrenament dels models realitzat amb èxit.")
                else:
                    print("Errors durant l'entrenament dels models.")
            # Control de múltiples excepcions (interrupció manual i error)
            except KeyboardInterrupt:
                logging.warning("Entrenament interromput manualment")
            except Exception as error:
                print(f"Error durant l'entrenament/calibració dels models: {error}")

        case "3":
            try:
                print("Iniciant l'anàlisi d'inferència per a tots els models")
                retorn_inferencia = _orquestrar_inferencia(models_datasets, dispositiu)
                # Notificar resultat de l'execució
                if retorn_inferencia:
                    print("Anàlisi d'inferència finalitzada amb èxit.")
                else:
                    print("Errors durant l'anàlisi d'inferència dels models.")
            # Control de múltiples excepcions (interrupció manual i error)
            except KeyboardInterrupt:
                logging.warning("Entrenament interromput manualment")
            except Exception as error:
                print(f"Error durant l'anàlisi d'inferència dels models: {error}")
        case _:
            print("Opció seleccionada incorrecta")


def _configurar_logging_model() -> None:
    """Configura el sistema de logs del model tant a consola com a un fitxer.

    Els logs s'emmagatzemaran tant en la consola d'execució com en un fitxer alhora. Hi
    haurà diferents nivells de missatges (INFO, ERROR, etc.) i l'estructura del format
    serà HORES:MINUTS:SEGONS - NIVELL MISSATGE - MISSATGE.
    """
    # Obtenció del possible logger d'anteriors models i eliminació dels handlers/outputs
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Creació del format del log i setejar el nivell a informació
    format_log = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    logger.setLevel(logging.INFO)

    # Assegurar que les carpetes de logs i evolució del nou model existeixen
    parametres.DIR_LOGS.mkdir(parents=True, exist_ok=True)
    parametres.DIR_EVOLUCIO.mkdir(parents=True, exist_ok=True)

    # Creació dels dos handlers (consola i fitxer) per gestionar els logs
    handler_consola = logging.StreamHandler(sys.stdout)
    handler_consola.setFormatter(format_log)
    handler_logs = logging.FileHandler(parametres.RUTA_FITXER_LOGS, encoding="UTF-8")
    handler_logs.setFormatter(format_log)

    # Afegir els dos handlers al logger per registrar els events
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


def _orquestrar_entrenament(
    models_datasets: list[dict[str, Any]], dispositiu: torch.device
) -> bool:
    """Orquestra l'entrenament dels models i també realitza la seva calibració.

    Funció que orquestra l'entrenament i calibració dels diferents models d'entrada
    mitjançant un bucle on es fa la divisió de les dades de cadascun i es realitza
    l'entrenament i calibració de seqüencialment. En cas de que hi hagi algu error
    al llarg del procés en qualsevol dels models, retornarà False per propagar-lo.

    Args:
        models_datasets: Llista dels models a entrenar i els seus datasets.
        dispositiu: Dispositiu (GPU o CPU) amb el que s'entrena el model.

    Returns:
        Booleà amb el retorn de l'execució (True si finalitza correctament).
    """
    # Variable de retorn per verificar la compleció de l'entrenament i calibració dels
    # models (es canviarà a False en cas d'error en qualsevol dels passos)
    models_completats_amb_exit = True

    # Bucle principal amb l'entrenament i calibració dels diferents models
    for model in models_datasets:
        # Actualització del nom del model i els seus datasets
        parametres.NOM_MODEL = str(model["nom"])
        parametres.DATASETS_ACTIUS = tuple(model["datasets"])

        # Configurar el log i dispositiu utilitzat
        _configurar_logging_model()
        _registrar_configuracio_model(dispositiu)

        try:
            # Dividir dades en els diferents conjunts d'entrenament, validacio i test
            logging.info(
                f"Verificant particions de dades del model {parametres.NOM_MODEL}."
            )
            divisio_dades()

            # Entrenament del model
            logging.info(f"Executant entrenament del model {parametres.NOM_MODEL}.")
            model_finalitzat = entrenament_model(dispositiu)

            # Verificar que ha finalitzat i, en tal cas, fer la calibració d'aquest
            if model_finalitzat:
                logging.info(
                    f"Entrenament del model {parametres.NOM_MODEL} completat amb èxit."
                )
                logging.info(f"Executant calibració del model {parametres.NOM_MODEL}.")
                # Calibració i verificació de la seva compleció
                calibracio_finalitzada = calibrar_model(dispositiu)
                if calibracio_finalitzada:
                    logging.info(
                        f"Calibració del model {parametres.NOM_MODEL} realitzada."
                    )
                else:
                    logging.info(
                        f"Calibració del model {parametres.NOM_MODEL} fallada."
                    )
                    models_completats_amb_exit = False
            else:
                logging.error(f"Entrenament del model {parametres.NOM_MODEL} fallat.")
                models_completats_amb_exit = False
        except Exception as error:
            logging.error(f"Error del model {parametres.NOM_MODEL}: {error}")
            models_completats_amb_exit = False

    # Retorn de la funció indicant si algun model ha fallat
    return models_completats_amb_exit


def _orquestrar_inferencia(
    models_datasets: list[dict[str, Any]], dispositiu: torch.device
) -> bool:
    """Orquestra l'anàlisi d'inferència dels models entrenats.

    Funció que orquestra l'anàlisi d'inferència dels diferents models d'entrada
    mitjançant un bucle on primerament es generen les mètriques (si existeix la
    calibració del model). Aquestes es faran a partir de tots els datasets no utilitzats
    durant l'entrenament (o de les mostres separades per cada dataset expressament). Tot
    seguit s'harmonitzen, es calcula els coeficients Dice (en el cas dels datasets amb
    tumors), es generen les estadístiques de significació i rendiment i, finalment, es
    generen les gràfiques necessàries. En cas de que hi hagi algu error al llarg del
    procés en qualsevol dels models, retornarà False per propagar-lo.

    Args:
        models_datasets: Llista dels models a entrenar i els seus datasets.
        dispositiu: Dispositiu (GPU o CPU) amb el que es fa la inferència del model.

    Returns:
        Booleà amb el retorn de l'execució (True si finalitza correctament).
    """
    # Variable de retorn per verificar la compleció de l'anàlisi d'inferència dels
    # models (es canviarà a False en cas d'error en qualsevol dels passos)
    models_completats_amb_exit = True

    # Recuperar els datasets de test de la configuració per cada model
    datasets_test_originals = parametres.DATASETS_TEST

    # Bucle principal de l'anàlisi
    for model in models_datasets:
        # Actualització del nom del model i dels datasets actius
        parametres.NOM_MODEL = str(model["nom"])
        parametres.DATASETS_ACTIUS = tuple(model["datasets"])

        # Obtenir els datasets de test originals per ampliar amb els principals
        llista_tests = list(datasets_test_originals)

        # Afegir els datasets principals d'entrenament (IXI, HCP o OASIS) al llistat de
        # datasets de test i actualitzar la variable de DATASETS_TEST
        for nom_dataset in parametres.DATASETS_ENTRENAMENT:
            if nom_dataset not in parametres.DATASETS_ACTIUS:
                llista_tests.append(nom_dataset)
        parametres.DATASETS_TEST = tuple(llista_tests)

        # Configurar el log i dispositiu utilitzat
        _configurar_logging_model()
        _registrar_configuracio_model(dispositiu)

        # Realitzar passos de l'anàlisi de cada model
        if parametres.RUTA_CALIBRACIO.exists():
            logging.info(f"Iniciant inferència del model {parametres.NOM_MODEL}.")
            try:
                # Generació de les mètriques
                logging.info("Generant mètriques.")
                if not generacio_metriques(dispositiu):
                    logging.error(
                        f"Error generant mètriques del model {parametres.NOM_MODEL}."
                    )
                    models_completats_amb_exit = False
                    continue

                # Harmonització mètriques
                logging.info("Harmonitzant mètriques del model.")
                aplicar_neurocombat()

                # Càlcul dels coeficients Dice
                logging.info("Calculant coeficients Dice pel dataset BRATS.")
                calcular_dice_brats(dispositiu)

                # Generar les estadístiques de significació i rendiment
                logging.info("Generant estadístiques de significació i rendiment.")
                llindar_95, llindar_youden = generar_estadistiques_i_rendiment()

                # Generar els gràfics de les mètriques del model
                logging.info("Generant gràfics de les mètriques del model.")
                generar_grafics(llindar_95, llindar_youden)

                logging.info(f"Inferència completada pel model {parametres.NOM_MODEL}.")

            except Exception as error:
                logging.error(
                    f"Error en la inferència del model {parametres.NOM_MODEL}: {error}"
                )
                models_completats_amb_exit = False
        else:
            logging.error(f"El model {parametres.NOM_MODEL} no té calibració.")
            models_completats_amb_exit = False
    return models_completats_amb_exit


# "Funció" principal per executar el codi
if __name__ == "__main__":
    executar_pipeline()
