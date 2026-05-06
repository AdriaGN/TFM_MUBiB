"""Mòdul encarregat d'avaluar totes les mostres de test i extreure les mètriques."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
from configuracio import parametres
from dades.carregador_dades import preparar_imatge_inferencia
from xarxa_neuronal.model import carregar_model_entrenat

from inferencia.calculador_incertesa import calcular_incertesa


def generacio_metriques(dispositiu: torch.device) -> bool:
    """Executa la inferència completa del model actiu i guarda els resultats.

    Primer es crea el directori d'inferència i s'obtenen tant les dades de test com del
    llindar del model. Tot seguit, per cada mostra, es processa i s'obté les diferents
    mètriques per emmagatzemar-les. A més, les 5 primeres mostres de cada dataset són
    exportades a NIfTI per documentar els resultats.

    Args:
        dispositiu: Dispositiu (GPU o CPU) amb el que es carrega el model.

    Returns:
        Booleà que retorna si les mètriques s'han o no generat correctament.
    """
    try:
        # Crear el directori d'inferència i obtenir la calibració, mostres test i model
        parametres.RUTA_INFERENCIA.mkdir(exist_ok=True, parents=True)
        llindar_calibracio, llista_arxius = _recopilar_dades_test()
        model = carregar_model_entrenat(dispositiu)

        # Bucle principal d'inferència per emplenar el CSV de mètriques
        resultats_csv = []
        for mostra in llista_arxius:
            # Obtenir mètriques i imatge transformada (amb la seva mitjana i variància)
            fila, imatge_transformada, mitjana, variancia = _processar_mostra(
                mostra, model, dispositiu, llindar_calibracio
            )
            resultats_csv.append(fila)

            # Generar NIfTIs de les 5 primeres mostres de cada dataset
            if mostra["Exportar_NIfTI"]:
                _exportar_exemples_nifti(
                    mostra,
                    imatge_transformada,
                    mitjana,
                    variancia,
                    llindar_calibracio,
                )

        # Emmagatzemar les mètriques no harmonitzades calculades del model
        pd.DataFrame(resultats_csv).to_csv(
            parametres.RUTA_METRIQUES, index=False, sep=";"
        )
        return True
    except Exception as error:
        print(f"Error durant la generació de mètriques: {error}")
        return False


def _recopilar_dades_test() -> Tuple[float, List[Dict[str, Any]]]:
    """Selecciona les mostres de test i les retorna junt amb el llindar de calibració.

    Comença llegint els fitxers de calibració i rutes de test reservat i extreu tant les
    mostres reservades com totes les que hi ha a les carpetes de test. A més, per cada
    conjunt, també selecciona les 5 primeres mostres de cada dataset per fer NIfTIs
    amb els resultats d'aplicar el model.

    Returns:
        El valor del llindar de calibració del model i la llista de mostres de test.
    """
    # Llegir els fitxers per obtenir la calibració i les mostres reservades per test
    with open(parametres.RUTA_CALIBRACIO, "r", encoding="utf-8") as f:
        llindar_calibracio = json.load(f)["Llindar_Model"]
    with open(parametres.RUTA_TESTS_RESERVATS, "r", encoding="utf-8") as f:
        test_reservats = json.load(f)

    # Crear llista buida a on emmagatzemar les mostres per fer la inferència
    mostres_test = []

    # Ampliar la llista amb les mostres dels fitxers de tests reservats amb un doble
    # bucle. El primer extreu el nom del dataset i les rutes i el segon itera les rutes
    # per triar les 5 primeres mostres (ordenades alfabèticament) per fer els NIfTIs
    for dataset, llista_rutes in test_reservats.items():
        for index_exportacio, ruta_mostra in enumerate(llista_rutes):
            # Castejar la ruta a Path per evitar problemes de format
            ruta = Path(ruta_mostra)
            # Afegir a la llista de mostres de test
            mostres_test.append(
                {
                    "Ruta": str(ruta.resolve()),
                    "Dataset": dataset,
                    "ID": ruta.name,
                    "Exportar_NIfTI": index_exportacio < 5,
                    "Index_Exportacio": index_exportacio + 1,
                }
            )

    # Repetir el mateix procés amb els arxius de la llista de datasets de test no
    # utilitzats durant l'entrenament (només utlitzant els fitxers NIfTIs ja ordenats)
    for dataset in parametres.DATASETS_TEST:
        dir_dataset = parametres.DIR_NET / dataset
        for index, ruta in enumerate(sorted(dir_dataset.glob("*.nii.gz"))):
            mostres_test.append(
                {
                    "Ruta": str(ruta.resolve()),
                    "Dataset": dataset,
                    "ID": ruta.name,
                    "Exportar_NIfTI": index < 5,
                    "Index_Exportacio": index + 1,
                }
            )

    return llindar_calibracio, mostres_test


def _processar_mostra(
    mostra: Dict[str, Any],
    model: nn.Module,
    dispositiu: torch.device,
    llindar_calibracio: float,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Executa la inferència d'una mostra i en calcula les mètriques d'avaluació.

    A partir de les metadades de la mostra, es carrega la imatge al dispositiu
    i es passa per la xarxa per obtenir-ne la reconstrucció i el mapa d'incertesa
    (variància). Tot seguit, es creua aquesta informació amb el diagnòstic clínic
    derivat del nom del dataset per generar un diccionari amb les mètriques clau.

    Args:
        mostra: "Metadades" de la mostra (ruta, dataset, etc.)
        model: Model de xarxa neuronal amb els millors pesos ja carregats.
        dispositiu: Dispositiu (GPU o CPU) amb el que es carrega el model.
        llindar_calibracio: Llindar de calibració del model.

    Returns:
        Mètriques i els tensors de la imatge transformada, la mitjana i variància.
    """
    # Extreure el nom del dataset i obtenir el seu diagnòstic
    dataset = mostra["Dataset"]
    patologic, cdr = _extreure_diagnostic_clinic(dataset)

    # Passar la imatge transformada pel model per obtenir la mitjana i variància
    imatge_transformada = preparar_imatge_inferencia(mostra["Ruta"], dispositiu)
    mitjana, variancia = calcular_incertesa(model, imatge_transformada)

    # Emplenar les metriques de la mostra amb els seus resultats.
    metriques_mostra = {
        "ID": mostra["ID"],
        "Dataset": dataset,
        "Nom_Dataset": dataset.split("_")[0],
        "Patologic": patologic,
        "CDR": cdr,
        "Pic_Intensitat_Maxima": variancia.max().item(),
        "Diferencia_Global_Mitjana": variancia.mean().item(),
        "Diferencia_Reconstruccio_L1": f.l1_loss(mitjana, imatge_transformada).item(),
        parametres.METRICA_TOTAL: int((variancia > llindar_calibracio).sum().item()),
    }

    return metriques_mostra, imatge_transformada, mitjana, variancia


def _extreure_diagnostic_clinic(origen_mostra: str) -> Tuple[int, str]:
    """Extreu el diagnòstic clínic (presència de patologia i CDR/Tumor) d'una mostra.

    La funció retorna (1, Tumor) en el cas de les mostres del dataset BraTS (ja que són
    sempre patològiques degut als tumors) o (0, 0.0) en el cas dels datasets sense "CDR"
    en el nom (ja que seran sempre sans). En el cas dels que tenen CDR, extreu el seu
    valor i el retorna com (1, CDR) si el CDR és superior a 0.0 o (0, 0.0) si és igual a
    0.0. Com que les imatges d'ADNI van per rangs de CDR, s'aproxima al CDR superior .

    Args:
        origen_mostra: String amb el nom de la mostra a analitzar.

    Returns:
        La presència o no de patologia (1 o 0) i el valor CDR o Tumor d'aquesta.
    """
    # Cas dataset BRATS (són sempre patològics)
    if "BRATS" in origen_mostra:
        return 1, "Tumor"

    # Cas datasets amb CDR (si és superior a 0.0 és patològic)
    if "CDR_" in origen_mostra:
        cdr = origen_mostra.split("CDR_")[1]
        cdr = {"0.5_A_1.0": "1.0", "2.0_A_3.0": "3.0"}.get(cdr, cdr)
        return int(cdr != "0.0"), cdr

    # Retorna cas no patològic
    return 0, "0.0"


def _exportar_exemples_nifti(
    mostra: Dict[str, Any],
    imatge: torch.Tensor,
    mitjana: torch.Tensor,
    variancia: torch.Tensor,
    llindar_calibracio: float,
) -> None:
    """Exporta els tensors resultants a format NIfTI per a l'avaluació visual.

    Primer de tot es crea la carpeta de cada imatge dintre de la carpeta de resultats
    d'inferència del model i s'obté la direccionalitat i mida de la imatge ja que,
    segons la documentació de Nibabel, la imatge pot quedar deformada sinó. Tot seguit
    es crea una llista amb els tensors i els noms dels NIfTIs a guardar i es transformen
    a matrius de numpy (en la CPU) per poder ser guardats per Nibabel.

    Args:
        mostra: Diccionari amb les metadades de la imatge (Ruta, Dataset, etc.).
        imatge: Tensor de la imatge original transformada.
        mitjana: Tensor de la imatge reconstruïda pel model.
        variancia: Tensor del mapa d'incertesa del model (variància).
        llindar_calibracio: Llindar de calibració del model.
    """
    # Crear el directori i el diccionari d'exportació de les imatges
    dir_pacient = (
        parametres.RUTA_NIFTI_METRIQUES
        / f"{mostra['Dataset']}_{mostra['Index_Exportacio']}"
    )
    dir_pacient.mkdir(parents=True, exist_ok=True)

    # Recuperar la orientació de la matriu de la imatge original per guardar la direcció
    orientacio = nib.load(mostra["Ruta"]).affine

    # Agrupar els tensors a exportar amb el seu nom de fitxer
    tensors_a_exportar = {
        "Original.nii.gz": imatge,
        "Reconstruccio_Model.nii.gz": mitjana,
        "Variancia_Model.nii.gz": variancia,
        "Anomalies_Detectades.nii.gz": (variancia > llindar_calibracio).float(),
    }

    # Transformar cada imatge a una matriu de numpy per guardar-la amb nibabel
    for nom, tensor in tensors_a_exportar.items():
        matriu_imatge = tensor[0, 0].detach().cpu().to(torch.float32).numpy()
        nib.save(nib.Nifti1Image(matriu_imatge, orientacio), dir_pacient / nom)
