"""Mòdul encarregat del disseny de la xarxa neuronal i de retornar-lo carregat."""

from typing import Any

import torch
import torch.nn as nn
from configuracio import parametres


class XarxaNeuronal(nn.Module):
    """Arquitectura de la xarxa neuronal del projecte.

    Classe que conté tant l'estructura de la xarxa neuronal com el seu flux d'execució.
    Es basa en 3 fases (codificador (4 capes), dropout i descodificador (4 capes)):
        1. Codificador: Redueix a la meitat la imatge d'entrada (duplicant el nombre
           de canals a partir dels canals de sortida de la primera), normalitza batch
           i aplica LeakyRelu per evitar neurones mortes si el vòxel és negre (0).
        2. Dropout: Aplicació del dropout a l'espai latent per evitar overfitting i
           poder generar múltiples imatges diferents per la fase de test final.
        3. Descodificador: Duplica la mida de la imatge d'entrada (dividint entre 2
           els canals en cada cas menys en la capa de sortida), normalitza i aplica ReLU
           estàndard perquè ha donat millors resultats que el LeakyReLU. També aplica
           dropouts en cada capa per millorar els resultats i reduïr l'efecte còpia.
    Totes els canals es troben parametritzats en base 2 a partir del valor indicat al
    fitxer de configuració del projecte, amb l'excepció del quart, on té una reducció
    de l'espai latent per forçar el model a aprendre i no realitzar còpies de cervells.
    El kernel de sortida (4) és més gran que el d'entrada (3) per evitar artefactes en
    el resultat final.
    """

    def __init__(self) -> None:
        """Inicialització de l'estructura de capes de la xarxa."""
        super().__init__()

        # Definició dels canals
        canal_1 = parametres.CANALS_BASE  # 16
        canal_2 = canal_1 * 2  # 32
        canal_3 = canal_1 * 4  # 64
        canal_4 = parametres.CANALS_LATENT  # 16

        # Fase 1: Codificador
        self.codificador = nn.Sequential(
            # Capa 1
            nn.Conv3d(parametres.ENTRADA, canal_1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(canal_1),
            nn.LeakyReLU(parametres.RATIO_RELU, inplace=True),
            # Capa 2
            nn.Conv3d(canal_1, canal_2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(canal_2),
            nn.LeakyReLU(parametres.RATIO_RELU, inplace=True),
            # Capa 3
            nn.Conv3d(canal_2, canal_3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(canal_3),
            nn.LeakyReLU(parametres.RATIO_RELU, inplace=True),
            # Capa 4
            nn.Conv3d(canal_3, canal_4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(canal_4),
            nn.LeakyReLU(parametres.RATIO_RELU, inplace=True),
        )

        # Fase 2: Aplicació del dropout a l'espai latent
        self.dropout = nn.Dropout(p=parametres.RATIO_DROPOUT_LATENT)

        # Fase 3: Descodificador
        self.descodificador = nn.Sequential(
            # Capa 1
            nn.ConvTranspose3d(canal_4, canal_3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(canal_3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=parametres.RATIO_DROPOUT),
            # Capa 2
            nn.ConvTranspose3d(canal_3, canal_2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(canal_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=parametres.RATIO_DROPOUT),
            # Capa 3
            nn.ConvTranspose3d(canal_2, canal_1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(canal_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=parametres.RATIO_DROPOUT),
            # Capa 4
            nn.ConvTranspose3d(
                canal_1, parametres.SORTIDA, kernel_size=4, stride=2, padding=1
            ),
        )

    # Funció per definir el procés d'execució del model
    def forward(self, tensor: torch.Tensor) -> Any:
        """Definició del flux de dades de la xarxa neuronal.

        Args:
            tensor: Tensor d'entrada en Format [Batch, Canals, Profunditat, Alt, Ample]

        Returns:
            Tensor de la imatge reconstruïda a partir del model.
        """
        # Fase 1: Codificador
        espai_latent = self.codificador(tensor)
        # Fase 2: Dropout
        espai_latent_dropout = self.dropout(espai_latent)
        # Fase 3: Descodificador
        reconstruccio = self.descodificador(espai_latent_dropout)
        return reconstruccio


def carregar_model_entrenat(dispositiu: torch.device) -> XarxaNeuronal:
    """Carrega el model amb els millors pesos de l'entrenament.

    Primer s'obté el model de la Xarxa neuronal i es mou al dispositiu, on es carregarà
    amb els millors pesos obtinguts durant l'entrenament en mode avaluació.

    Args:
        dispositiu: Dispositiu (GPU o CPU) amb el que es carrega el model.

    Returns:
        Model de la xarxa neuronal amb els pesos ja carregats.
    """
    model = XarxaNeuronal()
    model.to(dispositiu)
    model.load_state_dict(
        torch.load(
            parametres.RUTA_PESOS_MILLOR, map_location=dispositiu, weights_only=True
        )
    )
    model.eval()
    return model
