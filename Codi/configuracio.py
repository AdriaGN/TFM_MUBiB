"""Configuració global i centralitzada de totes les variables del projecte."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple  # Necessari per obtenir els datasets actuals dinàmicament


@dataclass(frozen=True)
class Configuracio:
    """Conté tots els paràmetres i rutes del projecte.

    Classe en d'instància global (es retorna una única instància no modificable (atribut
    frozen es troba a true)) amb tots els paràmetres, variables i rutes ja configurades.
    Com que les rutes en les dataclasses es calculen en el moment d'instanciar, es fa
    servir @property per definir les rutes dinàmiques i evitar problemes de lectura.
    """

    ### 1. Generació de models
    # Selecció dinàmics dels datasets per crear el model actual
    DATASETS_ACTIUS: Tuple[str, ...] = ("IXI",)

    # Nom del model a generar
    NOM_MODEL: str = "Model_IXI"

    ### 2. Rutes de directoris
    # Ruta arrel del repositori i directoris principals
    DIR_ARREL: Path = Path(__file__).resolve().parent.parent
    DIR_CODI: Path = DIR_ARREL / "Codi"
    DIR_DADES: Path = DIR_ARREL / "Dades"
    DIR_MODELS: Path = DIR_ARREL / "Models"

    # Rutes dels directoris de les mostres sense tractar (raw) i tractades (net)
    DIR_RAW: Path = DIR_DADES / "dades_raw"
    DIR_NET: Path = DIR_DADES / "dades_net"

    # Rutes dinàmiques del directori del model actual, divisions, pesos, logs i
    # evolució. Es deshabilita errors o avisos d'estil amb "noqa: D102, N802"
    @property
    def DIR_MODEL_ACTUAL(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODELS / self.NOM_MODEL

    @property
    def DIR_LOGS(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Logs"

    @property
    def DIR_TENSORBOARD(self) -> Path:  # noqa: D102, N802
        return self.DIR_LOGS / "TensorBoard"

    @property
    def DIR_EVOLUCIO(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Evolucio"

    @property
    def RUTA_FITXER_LOGS(self) -> Path:  # noqa: D102, N802
        return self.DIR_LOGS / "Entrenament.log"

    @property
    def RUTA_DIVISIONS(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Divisions.csv"

    @property
    def RUTA_PESOS_MILLOR(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Pesos_millor_model.pth"

    @property
    def RUTA_PESOS_ULTIM(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Pesos_ultim_model.pth"

    ### 3. Paràmetres del projecte
    # Preprocessament amb l'extracció de cervells
    RATIO_ENTRENAMENT: float = 0.7
    RATIO_VALIDACIO: float = 0.15

    # Model
    ENTRADA: int = 1
    SORTIDA: int = 1
    CANALS_BASE: int = 32
    RATIO_DROPOUT: float = 0.2
    RATIO_RELU: float = 0.2

    # DataLoaders
    MIDA_BATCH: int = 10
    NOMBRE_TREBALLADORS: int = 4

    # Entrenament
    TAXA_APRENENTATGE: float = 1e-4
    CAIGUDA_PES_ADAMW: float = 1e-4
    ITERACIONS_REDUCCIO: int = 10
    FACTOR_REDUCCIO: float = 0.5
    EPOQUES: int = 200
    PES_L1: float = 0.2
    PES_SSIM: float = 0.8


# Instància única que la resta de scripts importaran
parametres = Configuracio()
