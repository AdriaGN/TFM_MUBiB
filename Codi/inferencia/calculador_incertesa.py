"""Mòdul encarregat de calcular la incertesa d'una mostra."""

from typing import Tuple

import torch
import torch.nn as nn
from configuracio import parametres


def calcular_incertesa(
    model: nn.Module, imatge_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula incertesa d'una imatge i en retorna la seva mitjana i variància.

    S'inicia el model en mode avaluació i s'activa el dropout en les capes de test.
    Tot seguit, es realitza 50 prediccions (o el valor de la configuració) d'una mateixa
    imatge i es calcula la mitjana i variància entre elles per calcular la incertesa.

    Args:
        model: La xarxa neuronal ja carregada amb els pesos del millor "checkpoint".
        imatge_tensor: Tensor de la imatge a analitzar.

    Returns:
        Tupla amb dos tensors (un per la mitjana i un per la variància) de la imatge
    """
    # Posar el model en mode avaluació i activar les capes de dropout durant el test
    model.eval()
    model.apply(_activar_dropout_test)

    # Crear llista buida on posar les prediccions
    llista_prediccions = []

    # Generar les prediccions utilitzant torch.no_grad(), com en l'entrenament, per
    # indicar que és una validació/test. Utilitzar també AMP per reproduir el procés
    # fet durant l'entrenament.
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            for _ in range(parametres.PASSADES_MODEL):
                prediccio = model(imatge_tensor)
                llista_prediccions.append(prediccio)

    # Apilar les imatges en un sol tensor per calcular la mitjana i la variància
    tensor_apilat = torch.stack(llista_prediccions, dim=0)

    # Calcular la mitjana i variància (dimensió 0 perquè agrupi totes les passades en
    # una sola dimensió del tensor i unbiased=False per evitar que la fórmula divideixi
    # pel nombre de passades - 1 en comptes del nombre de passades, tal com diu
    # la documentació)
    mitjana = torch.mean(tensor_apilat, dim=0)
    variancia = torch.var(tensor_apilat, dim=0, unbiased=False)

    return mitjana, variancia


def _activar_dropout_test(model: nn.Module) -> None:
    """Activa les capes de dropout del model.

    Es recorre el model per buscar les capes amb "Dropout"" i s'activen per poder
    aplicar el dropout durant la fase de test.

    Args:
        model: Model de la Xarxa Neuronal entrenada.
    """
    if "Dropout" in model.__class__.__name__:
        model.train()
