"""Configuració global i centralitzada de totes les variables del projecte."""

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Tuple,
)


@dataclass
class Configuracio:
    """Conté tots els paràmetres i rutes del projecte.

    Classe en d'instància global (es retorna una única instància) amb tots els
    paràmetres, variables i rutes ja configurades. Com que les rutes en les dataclasses
    es calculen en el moment d'instanciar, es fa servir @property per definir les rutes
    dinàmiques i evitar problemes de lectura.
    """

    ### 1. Selecció dels datasets i nom de generació del model
    # Nom del model a entrenar o utilitzar pel test
    NOM_MODEL: str = "Model_IXI"

    # Selecció dinàmics dels datasets per crear el model actual
    DATASETS_ACTIUS: Tuple[str, ...] = ("IXI",)

    # Datasets totals del projecte
    DATASETS_PROJECTE: Tuple[str, ...] = ("IXI", "HCP", "OASIS", "ADNI", "BRATS")

    # Datasets d'entrenament del projecte
    DATASETS_ENTRENAMENT: Tuple[str, ...] = ("IXI", "HCP", "OASIS")

    # Selecció dels datasets per fer la inferència i test de resultats del model
    DATASETS_TEST: Tuple[str, ...] = (
        "ADNI_AD_CDR_0.0",
        "ADNI_AD_CDR_0.5_A_1.0",
        "ADNI_AD_CDR_2.0_A_3.0",
        "BRATS_TUMORS",
        "OASIS_AD_CDR_0.5",
        "OASIS_AD_CDR_1.0",
        "OASIS_AD_CDR_2.0",
        "OASIS_AD_CDR_3.0",
    )

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
    # evolució, etc. Es deshabilita errors o avisos d'estil amb "noqa: D102, N802"
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
    def RUTA_FITXER_LOGS(self) -> Path:  # noqa: D102, N802
        return self.DIR_LOGS / "Entrenament.log"

    @property
    def DIR_EVOLUCIO(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Evolucio"

    @property
    def RUTA_DIVISIONS(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Divisions.csv"

    @property
    def RUTA_PESOS_MILLOR(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Pesos_millor_model.pth"

    @property
    def RUTA_PESOS_ULTIM(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Pesos_ultim_model.pth"

    @property
    def RUTA_TESTS_RESERVATS(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Rutes_Tests_Reservats.json"

    @property
    def RUTA_CALIBRACIO(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Calibracio.json"

    @property
    def RUTA_METRIQUES(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Metriques_No_Harmonitzades.csv"

    @property
    def RUTA_HARMONITZACIO(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Metriques_Harmonitzades.csv"

    @property
    def RUTA_INFERENCIA(self) -> Path:  # noqa: D102, N802
        return self.DIR_MODEL_ACTUAL / "Resultats_Inferencia"

    @property
    def RUTA_ESTADISTIQUES(self) -> Path:  # noqa: D102, N802
        return self.RUTA_INFERENCIA / "Resultats_Analisi.txt"

    @property
    def RUTA_NIFTI_METRIQUES(self) -> Path:  # noqa: D102, N802
        return self.RUTA_INFERENCIA / "NIfTI_GENERATS"

    @property
    def RUTA_GRAFICS(self) -> Path:  # noqa: D102, N802
        return self.RUTA_INFERENCIA / "Grafics"

    ### 3. Paràmetres del projecte
    # Llavor reproductibilitat
    LLAVOR: int = 123

    # Preprocessament amb l'extracció de cervells
    RATIO_ENTRENAMENT: float = 0.7
    RATIO_VALIDACIO: float = 0.15

    # Model
    ENTRADA: int = 1
    SORTIDA: int = 1
    CANALS_BASE: int = 16
    CANALS_LATENT: int = 16
    RATIO_DROPOUT_LATENT: float = 0.15
    RATIO_DROPOUT: float = 0.05
    RATIO_RELU: float = 0.2

    # DataLoaders
    MIDA_BATCH: int = 10
    NOMBRE_TREBALLADORS: int = 4

    # Entrenament
    TAXA_APRENENTATGE: float = 2e-4
    CAIGUDA_PES_ADAMW: float = 1e-4
    ITERACIONS_REDUCCIO: int = 15
    FACTOR_REDUCCIO: float = 0.5
    EPOQUES: int = 200
    PES_L1: float = 0.2
    PES_SSIM: float = 0.8
    TALL_GRADIENTS: bool = True
    PES_GRADIENTS: float = 0.5
    FACTOR_SOROLL: float = 0.05

    # Test i mapes de calor
    PASSADES_MODEL: int = 50
    TALL_CALIBRACIO: int = 95
    Z_CALIBRACIO: float = 1.96
    MOSTRES_DATASET_CALIBRACIO: int = 10
    METRICA_TOTAL: str = "Volum_Anomalia_Total"
    METRIQUES_HARMONITZAR: Tuple[str, ...] = (
        "Pic_Intensitat_Maxima",
        "Diferencia_Global_Mitjana",
        "Diferencia_Reconstruccio_L1",
        METRICA_TOTAL,
    )


# Instància única que la resta de scripts importaran
parametres = Configuracio()
