"""Mòdul encarregat del disseny de la xarxa neuronal."""

from typing import Any  # Necessari per evitar problemes de retorn en forward()

import torch
import torch.nn as nn
from configuracio import parametres


class XarxaNeuronal(nn.Module):
    """Arquitectura de la xarxa neuronal del projecte.

    Classe que conté tant l'estructura de la xarxa neuronal com el seu flux d'execució.
    Es basa en 3 fases (codificador (4 capes), dropout i descodificador (4 capes)):
        1. Codificador: Redueix a la meitat la imatge d'entrada (duplicant el nombre
           de canals, a partir dels 32 de sortida de la primera), normalitza el batch
           i aplica LeakyRelu per evitar neurones mortes si el vòxel és negre (0).
        2. Dropout: Aplicació del dropout a l'espai latent per evitar overfitting i
           poder generar múltiples imatges diferents per la fase de test final.
        3. Descodificador: Duplica la mida de la imatge d'entrada (dividint entre 2
           els canals en cada cas menys la capa de sortida), normalitza i aplica ReLU
           estàndard perquè ha donat millors resultats que el LeakyReLU. Acaba amb una
           funció sigmoide per mantenir l'escala de grisos (entre 0 i 1).
    Totes els canals es troben parametritzats en base 2 a partir del valor indicat al
    fitxer de configuració del projecte. El kernel de sortida (4) és més gran que el
    d'entrada (3) per evitar artefactes en el resultat final.
    """

    def __init__(self) -> None:
        """Inicialització de l'estructura de capes de la xarxa."""
        super().__init__()

        # Definició dels canals
        canal_1 = parametres.CANALS_BASE  # 32
        canal_2 = canal_1 * 2  # 64
        canal_3 = canal_1 * 4  # 128
        canal_4 = canal_1 * 8  # 256

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
        self.dropout = nn.Dropout3d(p=parametres.RATIO_DROPOUT)

        # Fase 3: Descodificador
        self.descodificador = nn.Sequential(
            # Capa 1
            nn.ConvTranspose3d(canal_4, canal_3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(canal_3),
            nn.ReLU(inplace=True),
            # Capa 2
            nn.ConvTranspose3d(canal_3, canal_2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(canal_2),
            nn.ReLU(inplace=True),
            # Capa 3
            nn.ConvTranspose3d(canal_2, canal_1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(canal_1),
            nn.ReLU(inplace=True),
            # Capa 4
            nn.ConvTranspose3d(
                canal_1, parametres.SORTIDA, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
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
