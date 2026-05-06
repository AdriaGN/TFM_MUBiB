"""Mòdul encarregar de calcular el coeficient Dice dels datasets."""

import json

import nibabel as nib
import pandas as pd
import torch
from configuracio import parametres
from dades.carregador_dades import preparar_imatge_inferencia
from monai.metrics import compute_dice
from monai.transforms import ResizeWithPadOrCrop
from xarxa_neuronal.model import carregar_model_entrenat

from inferencia.calculador_incertesa import calcular_incertesa


def calcular_dice_brats(dispositiu: torch.device) -> None:
    """Calcula el coeficient Dice de les mostres del dataset BraTS.

    Primer s'obté el fitxer de mètriques, la calibració i les mostres de BraTS. Tot
    seguit, es carrega el model i es fa una superposició de les màscares predites i les
    reals amb compute_dice() de MONAI per obtenir el coeficient Dice de cadascuna. Per
    acabar, s'actualitza la columna de mètriques amb aquest valor.

    Args:
        dispositiu: Dispositiu (GPU o CPU) amb el que es carrega el model.
    """
    # Carregar mètriques harmonitzades, el llindar de calibració i seleccionar només les
    # mostres del dataset amb tumors (BraTS)
    metriques = pd.read_csv(parametres.RUTA_HARMONITZACIO, sep=";")
    with open(parametres.RUTA_CALIBRACIO, "r", encoding="utf-8") as f:
        llindar_calibracio = json.load(f)["Llindar_Model"]
    mostres_brats = metriques[metriques["Dataset"].str.contains("BRATS")]

    # Obtenir les rutes del dataset i les seves màscares
    directori_brats = parametres.DIR_NET / "BRATS_TUMORS"
    directori_mascares = directori_brats / "MASCARES"

    # Carregar el model i la transformació de mida de MONAI (com en l'entrenament)
    model = carregar_model_entrenat(dispositiu)
    transformacio_mida = ResizeWithPadOrCrop(spatial_size=(192, 224, 192))

    # Bucle principal de superposició de màscares i càlcul del coeficient Dice
    for id_mostra, fila in mostres_brats.iterrows():
        # Obtenir la ruta de la imatge i de la seva màscara del dataset
        ruta_imatge = directori_brats / fila["ID"]
        ruta_mascara = directori_mascares / f"mask_{fila['ID']}"

        try:
            # Obtenir la variància de la imatge calculada pel model
            imatge_transformada = preparar_imatge_inferencia(ruta_imatge, dispositiu)
            _, variancia = calcular_incertesa(model, imatge_transformada)

            # Transformar la variància en un mapa de 1 i 0 per indicar els punts predits
            tensor_prediccio = (variancia > llindar_calibracio).cpu().float()

            # Obrir la màscara original amb la llibreria nibabel i afegir el canal amb
            # unsqueeze per adaptar-ho al format de MONAI (tal com diu la documentació).
            # Després, aplicar la transformació de la mida per fer-la equivalent.
            mascara = torch.as_tensor(nib.load(ruta_mascara).get_fdata()).unsqueeze(0)
            mascara_transformada = transformacio_mida(mascara)

            # Afegir dimensió (com demana MONAI) triant només els voxels de la màscara
            tensor_real = (mascara_transformada > 0).unsqueeze(0).float()

            # Calcular Dice (incloent fons per assegurar el canal (segons documentació))
            resultat_dice = compute_dice(
                y_pred=tensor_prediccio, y=tensor_real, include_background=True
            )

            # Extreure el resultat del Dice i actualitzar la cel·la de la mostra
            metriques.at[id_mostra, "Coeficient_Dice"] = resultat_dice.item()

        except Exception as e:
            print(f"Error calculant el Dice de la mostra {fila['ID']}: {e}")

    # Sobreescriure el fitxer de metriques harmonitzades amb els coeficients Dice
    metriques.to_csv(parametres.RUTA_HARMONITZACIO, index=False, sep=";")
