"""Mòdul encarregat de fer les estadístiques de significació i rendiment del model."""

from typing import Tuple

import numpy as np
import pandas as pd
import pingouin as pg
from configuracio import parametres
from sklearn.metrics import auc, classification_report, roc_curve


def generar_estadistiques_i_rendiment() -> Tuple[float, float]:
    """Calcula el llindar del 95% (1.96 sigma) i de Youden .

    Primer es carrega el fitxer de les mètriques ja harmonitzades i es crea una carpeta
    a on deixar els resultats. Finalment, es calculen els nivells de significació del
    model i el seu rendiment (amb corbes ROC i la AUC) i es retorna el llindar del 95% i
    el de Youden.

    Returns:
        Tupla amb els llindars de Youden i del 95% per utilitzar en gràfics.
    """
    # Lectura del fitxer de mètriques harmonitzades
    metriques = pd.read_csv(parametres.RUTA_HARMONITZACIO, sep=";")

    # Crear una carpeta a on posar els resultats de l'anàlisi estadístic del model
    parametres.RUTA_ESTADISTIQUES.parent.mkdir(exist_ok=True)

    # Calcular mètriques de significació i rendiment
    _calcular_significacio(metriques)
    llindar_95, llindar_youden = _calcular_rendiment(metriques)

    return llindar_95, llindar_youden


def _calcular_significacio(metriques: pd.DataFrame) -> None:
    """Calcula els P-valors i desa un informe dels resultats estadístics.

    Comença netejant les dades i obtenint el subset de mostres sanes (CDR = 0) i fa un
    bucle per calcular per un U Test de Mann-Whitney per cada CDR. Finalment, genera un
    informe amb els resultats.

    Args:
        metriques: DataFrame que conté les mètrica harmonitzades del model.
    """
    # Eliminar CDR nuls i obtenir el grup de control (CDR = 0) per fer el U-Test
    cdr_net = metriques.dropna(subset=["CDR", parametres.METRICA_TOTAL]).copy()
    grup_control = cdr_net[cdr_net["CDR"] == "0.0"][parametres.METRICA_TOTAL]

    # Bucle d'execució on es fa el test per cadascun dels CDR
    llista_resultats = []
    for cdr in ["0.5", "1.0", "2.0", "3.0"]:
        # Seleccionar grup de CDR a testejar
        grup_test = cdr_net[cdr_net["CDR"] == cdr][parametres.METRICA_TOTAL]
        # Fer el U Test amb la funció mwu de la llibreia pingouin
        taula_u_test = pg.mwu(grup_control, grup_test, alternative="two-sided")
        # Afegir columnes per referenciar la comparativa i indicar la  mida i mediana
        taula_u_test.insert(0, "Comparativa", f"CDR 0.0 vs {cdr}")
        taula_u_test.insert(1, "Mida", len(grup_test))
        taula_u_test.insert(2, "Mediana", grup_test.median())
        llista_resultats.append(taula_u_test)

    # Concatenar tots els resultats en una sola taula
    taula_test = pd.concat(llista_resultats, ignore_index=True)

    # Crear els resultats estadístic amb capçalera per diferenciar-lo del de rendiment
    resultats = f"""{"=" * 50}\nResultats estadístics (Mann-Whitney U Test)\n
Grup control (CDR 0.0): Mida = {len(grup_control)}, Mediana = {grup_control.median()}\n
{taula_test.to_string(index=False)}\n{"=" * 50}\n"""

    # Guardar els resultats al fitxer
    with open(parametres.RUTA_ESTADISTIQUES, "w", encoding="utf-8") as f:
        f.write(resultats)


def _calcular_rendiment(metriques: pd.DataFrame) -> Tuple[float, float]:
    """Calcula el rendiment clínic del model i n'extreu els llindars de tall.

    Comença separant el dataset BraTS de la resta (perquè la seva avaluació és diferent
    ja que busca comparar àrees i està composat exclusivament per cervells patològics) i
    neteja les dades. Tot seguit obté el subset de mostres sanes, calcula la corba
    ROC, els llindars, l'àrea sota la corba (AUC), ho escriu en el fitxer anterior i
    acaba retornant els llindars per poder-los dibuixar més endavant.

    Args:
        metriques: DataFrame que conté les mètrica harmonitzades del model.

    Returns:
        Tupla amb els llindars del 95% i de Youden.
    """
    # Separació de les mostres del dataset BraTS per excloure-les de l'anàlisi final
    metriques = metriques[~metriques["Dataset"].str.contains("BRATS")]

    # Netejar dades eliminant valors nuls i agrupar-les per la mètrica d'avaluació
    # segons la presència d'alguna patologia
    dades_net = metriques.dropna(subset=[parametres.METRICA_TOTAL, "Patologic"])
    sans = dades_net[dades_net["Patologic"] == 0][parametres.METRICA_TOTAL]

    # Obtenir el Ratio de Falsos Positius (FPR), Positius Vertaders (TPR) i llindars
    fpr, tpr, llindars = roc_curve(
        dades_net["Patologic"], dades_net[parametres.METRICA_TOTAL]
    )

    # Calcular llindars ("clàssic" amb 95% (Z = 1.96) i òptim amb l'índex de Youden)
    llindar_95 = sans.mean() + (sans.std() * parametres.Z_CALIBRACIO)
    llindar_youden = llindars[np.argmax(tpr - fpr)]

    # Preparar els resultats de rendiment i estadístiques per escriure al fitxer
    resultats = f"""{"=" * 50}\nResultats rendiment\nLlindar 95%: {llindar_95}\nLlindar
 Youden: {llindar_youden}\nAUC: {auc(fpr, tpr)}\nResultats Llindar 95%:\n
{
        classification_report(
            dades_net["Patologic"], dades_net[parametres.METRICA_TOTAL] > llindar_95
        )
    }\n
Resultats Llindar Youden:\n{
        classification_report(
            dades_net["Patologic"], dades_net[parametres.METRICA_TOTAL] > llindar_youden
        )
    }
    """

    # Guardar els resultats al fitxer i retornar els llindars per dibuixar-los
    with open(parametres.RUTA_ESTADISTIQUES, "a", encoding="utf-8") as f:
        f.write(resultats)

    return float(llindar_95), float(llindar_youden)
