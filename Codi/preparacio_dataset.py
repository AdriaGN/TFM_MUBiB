"""Mòdul encarregat d'extraure els cervells de les mostres i de dividir el dataset."""

import csv
import random
import subprocess
from pathlib import Path
from typing import List

from configuracio import parametres


def _extraccio_cervells() -> None:
    """Utilitzar SynthStrip per aïllar els cervells de les mostres (Skull Stripping).

    L'extracció dels cervells es farà amb l'eina SynthStrip que es troba en un Docker.
    Primer de tot s'extreu les rutes dels datasets amb les mostres sense tractar (raw)
    i les dels cervells extrets (net) i es crea una llista de totes les mostres. Tot
    seguit, iterant per cadascuna d'aquestes mostres, es crida el Docker de SynthStrip
    com a subprocés i s'emmagatzema el cervell processat al director de cervells "nets"
    corresponent als repositoris actius a la configuració.
    """
    print(f"Inici d'extracció per a {len(parametres.DATASETS_ACTIUS)} datasets.")

    for nom_dataset in parametres.DATASETS_ACTIUS:
        ruta_raw = parametres.DIR_RAW / nom_dataset
        ruta_net = parametres.DIR_NET / nom_dataset
        # Creació del directori de sortida en cas de que no existeixi
        ruta_net.mkdir(parents=True, exist_ok=True)
        mostres_dataset = list(ruta_raw.glob("*.nii.gz"))

        print(f"Processant {len(mostres_dataset)} mostres del dataset {nom_dataset}.")
        for ruta_mostra in mostres_dataset:
            # Obtenció de la ruta de sortida de cada mostra netejada (fitxers net_XXX)
            ruta_sortida = ruta_net / f"net_{ruta_mostra.name}"

            # Execució de la comanda de Docker amb comprovació d'errors (check = True)
            if not ruta_sortida.exists():
                print(f"Processant la mostra {ruta_mostra.name}.")
                try:
                    subprocess.run(
                        [
                            "docker",
                            "run",
                            "--rm",
                            "-v",
                            f"{ruta_raw.resolve()}:/input",
                            "-v",
                            f"{ruta_net.resolve()}:/output",
                            "freesurfer/synthstrip:1.8",
                            "-i",
                            f"/input/{ruta_mostra.name}",
                            "-o",
                            f"/output/{ruta_sortida.name}",
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    print(f"Error processant la mostra {ruta_mostra.name}")


def _divisio_dades() -> None:
    """Divisió les dades pre-processades ("net") en diversos conjunts.

    La divisió parteix dels directoris amb els cervells ja nets per a cadascun dels
    diferents datasets actius i els afegeix a una llista conjunta anomenada
    "mostres_netes". Tot seguit es barregen i es divideix en els conjunts d'entrenament,
    validació i test seguint els ratios descrits en la configuració. Finalment, es crea
    un fitxer CSV en el directori del model per emmagatzemar els resultats i no haver de
    duplicar mostres en les diferents carpetes de cada model.

    """
    mostres_netes: List[Path] = []
    # Lectura de tots els datasets i creació de la llista conjunta de mostres netes
    for nom_dataset in parametres.DATASETS_ACTIUS:
        ruta_dataset = parametres.DIR_NET / nom_dataset
        if ruta_dataset.exists():
            mostres_dataset = list(ruta_dataset.glob("*.nii.gz"))
            mostres_netes.extend(mostres_dataset)

    if len(mostres_netes) != 0:
        # Divisió de les mostres en els conjunts d'entrenament, validació i test
        random.shuffle(mostres_netes)
        tall_1 = int(len(mostres_netes) * parametres.RATIO_ENTRENAMENT)
        tall_2 = tall_1 + int(len(mostres_netes) * parametres.RATIO_VALIDACIO)
        mostres_entrenament = mostres_netes[:tall_1]
        mostres_validacio = mostres_netes[tall_1:tall_2]
        mostres_test = mostres_netes[tall_2:]

        # Creació del directori de sortida del model en cas de que no existeixi
        parametres.DIR_MODEL_ACTUAL.mkdir(parents=True, exist_ok=True)

        print(
            f"Dividint les {len(mostres_netes)} mostres en {len(mostres_entrenament)}"
            f" mostres d'entrenament, {len(mostres_validacio)} mostres de validació i"
            f" {len(mostres_test)} mostres de test."
        )

        # Creació i escriptura en el fitxer CSV dels resultats de la divisió
        with open(
            parametres.RUTA_DIVISIONS, mode="w", newline="", encoding="UTF-8"
        ) as fitxer_csv:
            escriptor = csv.writer(fitxer_csv)
            # Afegir capçalera perquè Pandas ho pugui llegir
            escriptor.writerow(["divisio", "ruta_fitxer"])
            for mostra in mostres_entrenament:
                escriptor.writerow(["entrenament", str(mostra)])
            for mostra in mostres_validacio:
                escriptor.writerow(["validacio", str(mostra)])
            for mostra in mostres_test:
                escriptor.writerow(["test", str(mostra)])


if __name__ == "__main__":
    """Execució principal del programa per seleccionar quins tractaments cal aplicar."""
    # Mostrar les opcions i permetre a l'usuari triar-ne una
    print("1. Skull Stripping dels cervells")
    print("2. Generar particions Entrenament/Validació")
    print("3. Skull Stripping + Generar particions Entrenament/Validació/Test")
    opcio = input()

    # Switch-case amb les diferents opcions i retorn del seu estat a l'usuari
    match opcio:
        case "1":
            _extraccio_cervells()
            print("Skull Stripping feta.")
        case "2":
            if parametres.NOM_MODEL == "":
                print("Introduir nom del model a la configuració. Divisió no feta.")
            else:
                _divisio_dades()
                print("Divisió de les mostres feta.")
        case "3":
            _extraccio_cervells()
            if parametres.NOM_MODEL == "":
                print("Introduir nom del model a la configuració. Divisió no feta.")
            else:
                _divisio_dades()
                print("Skull Stripping i divisió de les mostres feta.")
        case _:
            print("Opció seleccionada incorrecta")
