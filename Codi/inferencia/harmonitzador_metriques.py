"""Mòdul que harmonitza les mètriques dels diferents datasets utilitzant NeuroCombat."""

import pandas as pd
from configuracio import parametres
from neuroCombat import neuroCombat


def aplicar_neurocombat() -> None:
    """Aplica una harmonització de les mètriques al fitxer de mètriques del model.

    Comença llegint el fitxer de mètriques originals i separa el datset BraTS de la
    resta ja que aquest està composat només amb mostres patològiques i trenca el procés
    d'harmonització (error de matriu singular). Tot seguit verifica que està format per
    més d'un dataset (ja que en cas contrari no es pot harmonitzar) i neteja les dades
    de valors nuls. Tot seguit transposa les dades i harmonitza aplicant NeuroCombat.
    Per acabar, guarda els resultats en el fitxer d'harmonització de la configuració
    (incloent les dades BraTS separades anteriorment).
    """
    # Lectura del CSV de mètriques original
    csv_original = pd.read_csv(parametres.RUTA_METRIQUES, sep=";")

    # Separar dades del dataset BraTS perquè són totes patològiques
    mascara_brats = csv_original["Dataset"].str.contains("BRATS")
    brats_apartats = csv_original[mascara_brats].copy()
    metriques_sense_brats = csv_original[~mascara_brats].copy()

    # Verificar que el model no està entrenat només per un sol dataset, ja que llavors
    # no es pot aplicar l'harmonització. Tot i això guarda l'original per futurs passos.
    if metriques_sense_brats["Dataset"].nunique() < 2:
        print("No cal aplicar NeuroCombat perquè el model el forma només un dataset.")
        csv_original.to_csv(parametres.RUTA_HARMONITZACIO, index=False, sep=";")
        return

    # Netejar columnes d'origen i patologia per assegurar que no hi hagi valors nuls
    columnes_neteja = list(parametres.METRIQUES_HARMONITZAR) + ["Dataset", "Patologic"]
    metriques_netes = metriques_sense_brats.dropna(subset=columnes_neteja).copy()

    # Transposar perquè NeuroCombat requereix [Features x Samples], segons documentació
    dades_transposades = metriques_netes[
        list(parametres.METRIQUES_HARMONITZAR)
    ].values.T

    # Execució de NeuroCombat amb les dades transposades, covariables, columna de lots
    # i la variable categorica a mantenir (Patologic)
    try:
        harmonitzacio = neuroCombat(
            dat=dades_transposades,
            covars=metriques_netes[["Nom_Dataset", "Patologic"]],
            batch_col="Nom_Dataset",
            categorical_cols=["Patologic"],
        )

        # Clonar el dataset original i actualitzar amb les mètriques harmonitzades
        metriques_harmonitzades = metriques_netes.copy()
        metriques_harmonitzades[list(parametres.METRIQUES_HARMONITZAR)] = harmonitzacio[
            "data"
        ].T

        # Concatenar de nou a les dades harmonitzades el conjunt BraTS separat abans
        metriques_harmonitzades = pd.concat(
            [metriques_harmonitzades, brats_apartats], ignore_index=True
        )

        # Guardar el nou dataset amb les harmonitzacions
        metriques_harmonitzades.to_csv(
            parametres.RUTA_HARMONITZACIO, index=False, sep=";"
        )
    # Control d'excepcions, guardant com abans el fitxer original per futurs passos.
    except Exception as e:
        print(f"Error durant l'harmonització: {e}.\nRetornat el CSV original")
        csv_original.to_csv(parametres.RUTA_HARMONITZACIO, index=False, sep=";")
