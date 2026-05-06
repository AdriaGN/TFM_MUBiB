"""Mòdul encarregat de calibrar un model a un llindar de normalitat global."""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from configuracio import parametres
from dades.carregador_dades import preparar_imatge_inferencia
from xarxa_neuronal.model import carregar_model_entrenat

from inferencia.calculador_incertesa import calcular_incertesa


def calibrar_model(dispositiu: torch.device) -> bool:
    """Calibra el model a partir dels paràmetres indicats a la configuració.

    Obté les rutes de calibració i calcula totes les variàncies de cadascuna d'elles per
    posteriorment fer el càlcul del llindar del model. Finalment, guarda aquesta
    calibració en un fitxer json.

    Args:
        dispositiu: Dispositiu (GPU o CPU) amb el que s'executa el model.

    Returns:
        Booleà que indica si s'ha calibrat correctament o no el model.
    """
    try:
        # Separació de les rutes de calibració i les reservades per testejar
        rutes_calibracio = _separar_test_calibracio()
        # Carregar model amb els pesos òptims
        model = carregar_model_entrenat(dispositiu)

        # Crear variable per guardar variàncies i calcular totes les de calibració
        variancies = []
        for llista_rutes in rutes_calibracio.items():
            for ruta in llista_rutes:
                tensor_imatge = preparar_imatge_inferencia(str(ruta), dispositiu)
                _, variancia = calcular_incertesa(model, tensor_imatge)
                variancies.append(variancia)

        # Calcular el llindar del model
        llindar_model = _calcular_percentil(variancies)

        # Crear un diccionari amb els resultats i emmagatzemar-lo en un fitxer json
        resultats = {
            "Nom_Model": parametres.NOM_MODEL,
            "Llindar_Model": llindar_model,
        }
        with open(parametres.RUTA_CALIBRACIO, "w", encoding="utf-8") as f:
            json.dump(resultats, f)

        # Logejar resultats i retornar True en cas que la calibració sigui correcta
        logging.info(f"Llindar del model {parametres.NOM_MODEL}: {llindar_model}")
        return True
    except Exception as e:
        # Logejar errors i retornar False en cas que la calibració sigui incorrecta
        logging.error(f"Error calibrant el model {parametres.NOM_MODEL}: {e}")
        return False


def _separar_test_calibracio() -> Dict[str, List[Path]]:
    """Llegeix el CSV de divisions i n'extreu un nombre fix de mostres per calibrar.

    Es selecciona les rutes de test del fitxer de divisions del model i s'agrupen segons
    el nom del dataset. Tot seguit, es fa una partició a l'atzar i s'aplica el punt de
    tall del fitxer de configuració. Finalment, es crida a la funció _guardar_rutes_test
    per guardar les rutes exclusivament de test i es retorna les de calibració.

    Returns:
        Diccionari amb els datasets i les seves rutes per fer la calibració del model.
    """
    # Llavor per reproductibilitat
    random.seed(parametres.LLAVOR)

    # Carregar en un dataframe les divisions de test i seleccionar-ne només les rutes
    divisions = pd.read_csv(parametres.RUTA_DIVISIONS)
    rutes_test = divisions[divisions["divisio"] == "test"]["ruta_fitxer"]

    # Agrupar les rutes dels datasets segons el seu nom
    diccionari_datasets = defaultdict(list)
    for ruta in rutes_test:
        diccionari_datasets[Path(ruta).parent.name].append(Path(ruta))

    # Creació de dues llistes buides per fer la separació de rutes en dos conjunts: un
    # de test i un de calibració
    rutes_calibracio = {}
    rutes_reservades = {}

    # Separar les rutes de cada dataset a l'atzar en les dues particions
    for dataset, llista_rutes in diccionari_datasets.items():
        # Divisió a l'atzar de les rutes
        random.shuffle(llista_rutes)
        # Aplicar particions a la llista de rutes
        rutes_calibracio[dataset] = llista_rutes[
            : parametres.MOSTRES_DATASET_CALIBRACIO
        ]
        rutes_reservades[dataset] = llista_rutes[
            parametres.MOSTRES_DATASET_CALIBRACIO :
        ]

    # Guardar les rutes de test reservades en un json per la fase de test
    _guardar_rutes_test(rutes_reservades)

    return rutes_calibracio


def _guardar_rutes_test(rutes_reservades: Dict[str, List[Path]]) -> None:
    """Guarda les rutes reservades exclusivament per fer el test del model.

    Transforma el conjunt de rutes (Path) a strings i les emmagatzema en un json.

    Args:
        rutes_reservades: Datasets i rutes de les imatges d'ús exclusiu durant el test.
    """
    rutes = {}
    # Convertir les rutes reservades de cada dataset en string per poder-les guardar
    for nom_dataset, llista_rutes in rutes_reservades.items():
        rutes[nom_dataset] = [str(ruta) for ruta in llista_rutes]

    # Guardar la llista de rutes transformades en un json
    with open(parametres.RUTA_TESTS_RESERVATS, mode="w", encoding="utf-8") as f:
        json.dump(rutes, f)


def _calcular_percentil(llista_variancies: List[torch.Tensor]) -> float:
    """Converteix els tensors 3D en un vector 1D i calcula el percentil estadístic.

    Primer de tot es redueix la dimensionalitat de cada tensor per poder-los concatenar
    en un array i després es fa el càlcul del valor percentil (indicat a configuració)
    que servirà com a valor per la calibració. En la primera part, s'utilitza .cpu() per
    poder utilitzar la lliberia NumPy correctament, ja que dona errors si s'utilitza en
    la GPU (tal i com indica la seva documentació).

    Args:
        llista_variancies: Llista de tensors amb totes les variàncies calculades.

    Returns:
        Llindar calculat del model a partir del percentil indicat en la configuració,
    """
    variancies = np.concatenate(
        [tensor.flatten().cpu().numpy() for tensor in llista_variancies]
    )
    return float(np.percentile(variancies, parametres.TALL_CALIBRACIO))
