"""Mòdul encarregat de dibuixar les gràfiques de les mètriques i els mapes de calor."""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from configuracio import parametres
from sklearn.metrics import auc, roc_curve


def generar_grafics(llindar_95: float, llindar_youden: float) -> None:
    """'Orquestrador' intern que genera tots els gràfics per fer l'anàlisi del model.

    Primer de tot es crea la ruta de la carpeta amb els gràfics i s'obtenen tant les
    mètriques "crues" (sense harmonitzar) com les harmonitzades. Tot seguit, es va
    cridant a la resta de funcions del mòdul per generar les mètriques.

    Args:
        llindar_95: Llindar del 95% calculat durant l'anàlisi estadística.
        llindar_youden: Llindar de Youden calculat durant l'anàlisi estadística.
    """
    # Crear el directori per guardar les gràfiques i llegir els dos fitxers de mètriques
    parametres.RUTA_GRAFICS.mkdir(parents=True, exist_ok=True)
    metriques_crues = pd.read_csv(Path(parametres.RUTA_METRIQUES), sep=";")
    metriques_harmonitzades = pd.read_csv(Path(parametres.RUTA_HARMONITZACIO), sep=";")

    # Dibuixar gràfics de dispersió abans i després d'harmonitzar
    _dibuixar_dispersio(metriques_crues, "Cru")
    _dibuixar_dispersio(metriques_harmonitzades, "Harmonitzat")

    # Dibuixar gràfic amb les corbes ROC (tant original com harmonitzada) i punt Youden
    _dibuixar_roc(metriques_crues, metriques_harmonitzades, llindar_youden)

    # Dibuixar diagrama de caixes amb l'efecte de l'harmonització
    _dibuixar_harmonitzacio(metriques_crues, metriques_harmonitzades)

    # Dibuixar diagrama de caixes amb la progressió del CDR (tant harmonitzat com no)
    _dibuixar_cdr(metriques_crues, llindar_95, llindar_youden, "Cru")
    _dibuixar_cdr(metriques_harmonitzades, llindar_95, llindar_youden, "Harmonitzat")

    # Dibuixar diagrama de caixes amb la comparació del volum d'anomalia total
    _dibuixar_comparativa_malaltia(metriques_crues, "Cru")
    _dibuixar_comparativa_malaltia(metriques_harmonitzades, "Harmonitzat")

    # Dibuixar histograma amb els valors agrupats del coeficient Dice del dataset BraTS
    _dibuixar_distribucio_dice(metriques_harmonitzades)

    # Generar mapes de calor 2D per fer comparacions
    _generar_mapes_calor()


def _dibuixar_dispersio(metriques: pd.DataFrame, titol: str) -> None:
    """Dibuixa un gràfic de dispersió dels datasets i de la seva patologia.

    Es comença dibuixant el gràfic de dispersió, s'etiqueta els eixos, es crea la
    llegenda i es guarda la imatge en alta qualitat.

    Args:
        metriques: Mètriques a dibuixar. Poden ser tant "crues" com harmonitzades.
        titol: Tipus de mètrica ("crua" o harmonitzada) per guardar les gràfiques.
    """
    # Dibuixar la gràfica de dispersió amb Seaborn
    sns.scatterplot(
        data=metriques,
        x="Diferencia_Global_Mitjana",
        y="Pic_Intensitat_Maxima",
        hue="Patologic",
        style="Nom_Dataset",
        palette={0: "palegreen", 1: "lightcoral"},
        alpha=0.75,
    )
    # Posar noms als eixos i crear llegenda
    plt.xlabel("Diferència Mitjana d'Anomalia")
    plt.ylabel("Pic d'Anomalia")
    punts, etiquetes_dataset = plt.gca().get_legend_handles_labels()
    plt.legend(punts, etiquetes_dataset, bbox_to_anchor=(1.01, 1.01))

    # Guardar la gràfica en format estret (per no perdre informació) i tancar PyPlot
    plt.tight_layout()
    plt.savefig(parametres.RUTA_GRAFICS / f"Grafic_Dispersio_{titol}.png", dpi=300)
    plt.close()


def _dibuixar_roc(
    metriques_crues: pd.DataFrame,
    metriques_harmonitzades: pd.DataFrame,
    llindar_youden: float,
) -> None:
    """Genera un gràfic amb dues corbes ROC ("crua" i harmonitzada) i el punt de Youden.

    Es comença calculant les corbes ROC de les mètriques harmonitzades i les "crues" i,
    tot seguit, el punt Youden. Finalment, s'afegeixen tots tres en el mateix gràfic, es
    dibuixa la linia de tall al 0.5, s'etiqueten els eixos, es crea la llegenda i es
    guarda la imatge en alta qualitat.

    Args:
        metriques_crues: Mètriques "crues" a dibuixar.
        metriques_harmonitzades: Mètriques harmonitzades a dibuixar.
        llindar_youden: Llindar de Youden calculat prèviament en l'anàlisi estadística.
    """
    # Obtenir el Ratio de Falsos Positius (FPR), Positius Vertaders (TPR) de cada grup
    # mètriques i calcular el punt de Youden
    fpr_cru, tpr_cru, _ = roc_curve(
        metriques_crues["Patologic"], metriques_crues[parametres.METRICA_TOTAL]
    )
    fpr_harmonitzat, tpr_harmonitzat, llindar_harmonitzat = roc_curve(
        metriques_harmonitzades["Patologic"],
        metriques_harmonitzades[parametres.METRICA_TOTAL],
    )
    punt_youden = np.argmin(np.abs(llindar_harmonitzat - llindar_youden))

    # Dibuixar la corba ROC de les mètriques crues
    plt.plot(
        fpr_cru,
        tpr_cru,
        label=f"ROC crua (AUC: {auc(fpr_cru, tpr_cru):.3f})",
    )

    # Dibuixar la corba ROC de les mètriques harmonitzades
    plt.plot(
        fpr_harmonitzat,
        tpr_harmonitzat,
        label=f"ROC harmonitzada (AUC: {auc(fpr_harmonitzat, tpr_harmonitzat):.3f})",
    )

    # Dibuixar el punt de Youden damunt de la corba ROC de la mètrica harmonitzada
    plt.scatter(
        fpr_harmonitzat[punt_youden],
        tpr_harmonitzat[punt_youden],
        color="red",
        zorder=2,
        label="Punt de Youden",
    )

    # Dibuixar la linia de tall de 0.5
    plt.plot([0, 1], [0, 1], linestyle=":")

    # Posar noms als eixos i crear llegenda
    plt.xlabel("Taxa de Falsos Positius")
    plt.ylabel("Taxa de Vertaders Positius")
    plt.legend(loc="lower right")

    # Guardar la gràfica generada i tancar PyPlot
    plt.savefig(parametres.RUTA_GRAFICS / "Comparativa_ROC.png", dpi=300)
    plt.close()


def _dibuixar_harmonitzacio(
    metriques_crues: pd.DataFrame, metriques_harmonitzades: pd.DataFrame
) -> None:
    """Dibuixa una gràfica que permet visualitzar l'efecte de l'harmonització.

    Inicialment es crea un dataset compartit amb les mètriques de les mostres sanes de
    les mètriques "crues" i harmonitzades (com requereix Seaborn, segons documentació).
    Tot seguit es crea el diagrama de parelles de caixes, es posa noms als eixos i es
    guarda la gràfica en alta qualitat.

    Args:
        metriques_crues: Mètriques "crues" a dibuixar.
        metriques_harmonitzades: Mètriques harmonitzades a dibuixar.
    """
    # Crear un dataset compartit per poder mostrar de costat els resultats de cada
    # dataset. Es seleccionen les mostres sense patologies i es crea una nova columna
    # amb el seu "estat" (harmonització aplicada o no)
    dataset_compartit = pd.concat(
        [
            metriques_crues[metriques_crues["Patologic"] == 0].assign(Estat="Cru"),
            metriques_harmonitzades[metriques_harmonitzades["Patologic"] == 0].assign(
                Estat="Harmonitzat"
            ),
        ]
    )

    # Dibuix del diagrama de caixes (en parelles segons dataset) amb Seaborn
    sns.boxplot(
        data=dataset_compartit,
        x="Nom_Dataset",
        y="Diferencia_Global_Mitjana",
        hue="Estat",
        palette={"Cru": "lightcoral", "Harmonitzat": "palegreen"},
    )

    # Posar noms als eixos
    plt.xlabel("Dataset")
    plt.ylabel("Error de Reconstrucció Mitjà")

    # Guardar la gràfica en format estret (per no perdre informació) i tancar PyPlot
    plt.tight_layout()
    plt.savefig(
        parametres.RUTA_GRAFICS / "Resultats_Harmonitzacio.png",
        dpi=300,
    )
    plt.close()


def _dibuixar_cdr(
    metriques: pd.DataFrame, llindar_95: float, llindar_youden: float, titol: str
) -> None:
    """Crea una gràfica que compara els volums d'anomalia total dels diferents CDR.

    Com a primer pas s'obtenen les mètriques dels CDR i es dibuixen en un diagrama de
    caixes amb una escala de groc a vermell. Tot seguit es dibuixen dues linies amb els
    llindars del 95% i el de Youden, es posa noms als eixos i es guarda la imatge.

    Args:
        metriques: Mètriques a dibuixar. Poden ser tant "crues" com harmonitzades.
        llindar_95: Llindar del 95% calculat prèviament en l'anàlisi estadística.
        llindar_youden: Llindar de Youden calculat prèviament en l'anàlisi estadística.
        titol: Tipus de mètrica ("crua" o harmonitzada) per guardar les gràfiques.
    """
    # Obtenir les mètriques amb CDR i convertir-les a strings per posar-les a la gràfica
    metriques_cdr = metriques.dropna(subset=["CDR"])
    metriques_cdr["CDR"] = metriques_cdr["CDR"].astype(str)

    # Dibuixar el diagrama de caixes agrupant pels diferents CDR analitzats (indicant
    # gravetat amb una escala de groc a vermell)
    sns.boxplot(
        data=metriques_cdr,
        x="CDR",
        y=parametres.METRICA_TOTAL,
        palette="YlOrRd",
        order=[0.0, 0.5, 1.0, 2.0, 3.0],
    )

    # Dibuixar una linia horitzontal amb el llindar del 95% i una amb el de Youden
    plt.axhline(
        llindar_95,
        color="lightcoral",
        label=f"Llindar 95% ({llindar_95:.0f})",
    )
    plt.axhline(
        llindar_youden,
        color="palegreen",
        label=f"Llindar Youden ({llindar_youden:.0f})",
    )

    # Posar noms a l'eix i crear la llegenda
    plt.ylabel("Volum d'Anomalia Global")
    plt.legend()

    # Guardar la gràfica i tancar PyPlot
    plt.savefig(
        parametres.RUTA_GRAFICS / f"Progressio_CDR_{titol}.png",
        dpi=300,
    )
    plt.close()


def _dibuixar_comparativa_malaltia(metriques: pd.DataFrame, titol: str) -> None:
    """Crea un diagrama de caixes amb el volum d'anomalia total de les malalties.

    Primer de tot es crea una nova columna amb el tipus de malaltia (segons si es
    compleixen unes certes condicions) o amb el valor sa. Tot seguit, es dibuixa
    un diagrama de caixes, es posa nom als eixoos i es guarda la gràfica en alta
    qualitat.

    Args:
        metriques: Mètriques a dibuixar. Poden ser tant "crues" com harmonitzades.
        titol: Tipus de mètrica ("crua" o harmonitzada) per guardar les gràfiques.
    """
    # Crear condicions datasets patològics amb tumors (BRATS) i Alzheimer (X.0) al nom
    condicions = [
        metriques["Dataset"].str.contains("0.5|1.0|2.0|3.0", regex=True),
        metriques["Dataset"].str.contains("BRATS"),
    ]

    # Assignació en una nova columna de la malaltia (sa per defecte)
    dataset_amb_malalties = metriques.assign(
        Malaltia=np.select(condicions, ["Alzheimer", "Tumor"], default="Sa")
    )

    # Dibuixar diagrama de caixes amb malalties sense llegenda (gràfic autoexplicatiu)
    sns.boxplot(
        data=dataset_amb_malalties,
        x="Malaltia",
        y=parametres.METRICA_TOTAL,
        hue="Malaltia",
        order=list({"Sa", "Alzheimer", "Tumor"}),
        palette={"palegreen", "lightblue", "lightcoral"},
        legend=False,
    )

    # Posar noms als eixos
    plt.xlabel("Grup Patològic")
    plt.ylabel("Volum d'Anomalia Global")

    # Guardar la gràfica en format estret (per no perdre informació) i tancar PyPlot
    plt.tight_layout()
    plt.savefig(parametres.RUTA_GRAFICS / f"Comparativa_Malalties_{titol}.png", dpi=300)
    plt.close()


def _dibuixar_distribucio_dice(metriques: pd.DataFrame) -> None:
    """Dibuixa un histograma amb la distribució dels coeficients Dice d'un model.

    Primer s'obtenen només els valors de la columna Dice (que no siguin nuls) i es crea
    un histograma a partir d'ells. Tot seguit s'afegeix una linia vertical amb la seva
    mitjana, es creen els eixos i es guarda la figura en alta qualitat.

    Args:
        metriques: Mètriques a dibuixar. Poden ser tant "crues" com harmonitzades.
    """
    # Obtenir coeficients Dice per fer la gràfica
    dataset_dice = metriques.dropna(subset=["Coeficient_Dice"])

    # Crear gràfica amb els coeficients Dice agrupats en grups de 20
    sns.histplot(dataset_dice["Coeficient_Dice"], bins=20, color="palegreen")

    # Dibuixar linia vertical amb la mitjana dels coeficients Dice
    plt.axvline(
        dataset_dice["Coeficient_Dice"].mean(),
        color="lightcoral",
        linestyle="-",
        label=f"Mitjana: {dataset_dice['Coeficient_Dice'].mean():.3f}",
    )

    # Posar noms al eixos i crear la llegenda
    plt.xlabel("Coeficient Dice")
    plt.ylabel("Nombre Mostres")
    plt.legend()

    # Guardar la gràfica i tancar PyPlot
    plt.savefig(parametres.RUTA_GRAFICS / "Distribucio_Dice.png", dpi=300)
    plt.close()


def _generar_mapes_calor() -> None:
    """Genera els mapes de calor per poder afegir als informes i comparacions finals.

    Obté els NIfTI originals, reconstruits i de variància generats anteriorment i fa una
    selecció del tall central de cadascun. Tot seguit, crea una imatge amb la original a
    l'esquerra, la reconstruida a la dreta i la superposició de la variància a la
    original a la dreta (semi-transparent i com a mapa de calor/"hot") Finalment afegeix
    una barra de colors/llegenda i guarda la imatge en alta qualitat.
    """
    # Obtenir els NIfTI generats durant fases anteriors i carregar les imatges
    for directori_nifti in parametres.RUTA_NIFTI_METRIQUES.iterdir():
        ruta_original = directori_nifti / "Original.nii.gz"
        ruta_reconstruccio = directori_nifti / "Reconstruccio_Model.nii.gz"
        ruta_variancia = directori_nifti / "Variancia_Model.nii.gz"

        imatge_original = nib.load(ruta_original).get_fdata()
        imatge_reconstruida = nib.load(ruta_reconstruccio).get_fdata()
        imatge_variancia = nib.load(ruta_variancia).get_fdata()

        # Seleccionar el tall central i rotar-lo 90º per seguir el format RAS
        tall_z = imatge_original.shape[2] // 2
        tall_original = np.rot90(imatge_original[:, :, tall_z])
        tall_reconstruit = np.rot90(imatge_reconstruida[:, :, tall_z])
        tall_variancia = np.rot90(imatge_variancia[:, :, tall_z])

        # Crear una imatge amb 3 imatges (original a l'esquerra, reconstrucció al centre
        # i superposició de la variància (en vermell) amb la original a la dreta)
        figura, grafics = plt.subplots(1, 3, figsize=(15, 5))

        # Imatge original
        grafics[0].imshow(tall_original, cmap="gray")
        grafics[0].set_title("Original")
        grafics[0].axis("off")

        # Imatge reconstruïda
        grafics[1].imshow(tall_reconstruit, cmap="gray")
        grafics[1].set_title("Reconstrucció")
        grafics[1].axis("off")

        # Superposició de la variància (vermell semi-transparent) a la imatge original
        grafics[2].imshow(tall_original, cmap="gray", alpha=0.5)
        superposicio_colors = grafics[2].imshow(
            tall_variancia, cmap="hot", alpha=0.75, vmin=0, vmax=1
        )
        grafics[2].set_title("Mapa de Calor")
        grafics[2].axis("off")

        # Dibuixar la barra/llegenda de colors de la superposició de la tercera imatge
        figura.colorbar(superposicio_colors)

        # Afegir titol amb el NIfTI comparat
        plt.suptitle(f"Comparació Resultats - {directori_nifti.name}", weight="bold")

        # Guardar la comparació en format estret i tancar PyPlot
        plt.tight_layout()
        plt.savefig(
            parametres.RUTA_GRAFICS / f"Mapa_Calor_{directori_nifti.name}.png",
            dpi=300,
        )
        plt.close()
