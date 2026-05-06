"""Mòdul encarregat de la càrrega i transformació dels fitxers NIfTI."""

from typing import Dict, List, Tuple  # Necessàri pels diferents "castings"

import pandas as pd
import torch
from configuracio import parametres
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
)


def obtenir_dataloaders_entrenament() -> Tuple[DataLoader, DataLoader]:
    """Retorna els dataloaders d'entrenament amb les dades d'entrenament i validació.

    Llegeix el fitxer Divisions.csv de la carpeta del model actual per obtenir les rutes
    assignades a les mostres d'entrenament i de validació, aplica les transformacions
    necessàries amb MONAI i crea dos DataLoaders (un d'entrenament i un de validació)
    per subministrar les dades a la GPU durant l'entrenament.

    Returns:
        Tupla amb el DataLoader d'entrenament i de validació.
    """
    # Verificar que el fitxer de divisions existeix prèviament
    if not parametres.RUTA_DIVISIONS.exists():
        raise FileNotFoundError("El fitxer de particions del model no existeix.")

    # Obtenció de les dades i filtratge de la divisió en els dos grups corresponents
    divisions = pd.read_csv(parametres.RUTA_DIVISIONS)
    rutes_entrenament = divisions[divisions["divisio"] == "entrenament"][
        "ruta_fitxer"
    ].tolist()
    rutes_validacio = divisions[divisions["divisio"] == "validacio"][
        "ruta_fitxer"
    ].tolist()

    # Convertir les llistes al format requerit per MONAI
    dades_entrenament = _crear_llista_diccionaris(rutes_entrenament)
    dades_validacio = _crear_llista_diccionaris(rutes_validacio)

    # Aplicar transformacions amb MONAI
    transformacions = _aplicar_transformacions()

    # Generació dels dos DataLoaders
    loader_entrenament = DataLoader(
        CacheDataset(
            data=dades_entrenament,
            transform=transformacions,
            cache_rate=1.0,
            num_workers=parametres.NOMBRE_TREBALLADORS,
        ),
        batch_size=parametres.MIDA_BATCH,
        shuffle=True,
        pin_memory=True,
    )

    loader_validacio = DataLoader(
        CacheDataset(
            data=dades_validacio,
            transform=transformacions,
            cache_rate=1.0,
            num_workers=parametres.NOMBRE_TREBALLADORS,
        ),
        batch_size=parametres.MIDA_BATCH,
        shuffle=False,
        pin_memory=True,
    )

    return loader_entrenament, loader_validacio


def preparar_imatge_inferencia(
    ruta_fitxer: str, dispositiu: torch.device
) -> torch.Tensor:
    """Prepara un fitxer NIfTI per ser processat pel model durant la fase de test.

    Aplica les transformacions de MONAI a una imatge i converteix la sortida a un tensor
    per enviar-lo a la GPU durant la fase de test. Cal utilitzar unsqueeze per assegurar
    que el tensor té les dimensions correctes (segons documentació PyTorch).

    Args:
        ruta_fitxer: Ruta del fitxer NIfTI a transformar.
        dispositiu: Dispositiu (GPU) a on enviar el tensor.

    Returns:
        Tensor ja transformat i preparat per passar al model durant el test.
    """
    # Generar diccionaris i aplicar les transformacions
    diccionari_entrada = _crear_llista_diccionaris([ruta_fitxer])[0]
    diccionari_transformat = _aplicar_transformacions()(diccionari_entrada)

    # Convertir a un tensor, aplicar unsqueeze i enviar al dispositiu
    return torch.as_tensor(diccionari_transformat["imatge"]).unsqueeze(0).to(dispositiu)


def _crear_llista_diccionaris(llista_rutes: List[str]) -> List[Dict[str, str]]:
    """Cerca fitxers NIfTI i genera la llista de diccionaris compatible amb MONAI.

    Args:
        llista_rutes: Llista de cadenes amb les rutes dels fitxers NIfTI a tractar.

    Returns:
        Llista de diccionaris on cada entrada segueix {"imatge": "ruta_NIfTI"}.
    """
    return [{"imatge": str(ruta)} for ruta in llista_rutes]


def _aplicar_transformacions() -> Compose:
    """Funció auxiliar que defineix i retorna les transformacions aplicades als NIfTI.

    La llista de transformacions aplicades, en ordre, és:
        1. Càrrega de la imatge
        2. Format de canals per assegurar que la dimensió del canal sigui la primera,
           tal i com requereix Conv3d de PyTorch.
        3. Normalització de la mida dels vòxels per uniformar mides dels datasets
        4. Normalització d'orientació per garantir que el NIfTI segueix l'estàndard RAS
           (Right, Anterior, Superior) i no tenir orientacions diferents.
        5. Retallar el fons per aïllar només el cervell.
        6. Normalitzar la intensitat de les imatges del model.
        8. Redimensionar la resolució a [192, 224, 192] per evitar problemes de memòria.

    Returns:
        Objecte compost de MONAI amb la seqüència de transformacions a aplicar.
    """
    return Compose(
        [
            LoadImaged(keys=["imatge"]),
            EnsureChannelFirstd(keys=["imatge"]),
            Spacingd(keys=["imatge"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["imatge"], axcodes="RAS"),
            CropForegroundd(keys=["imatge"], source_key="imatge"),
            NormalizeIntensityd(keys=["imatge"], nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=["imatge"], spatial_size=(192, 224, 192)),
        ]
    )
