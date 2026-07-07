"""
Configuración centralizada del proyecto.
Rutas, constantes y parámetros globales.
"""
import os
from pathlib import Path

# ─── Rutas base ───────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS_DIR = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"

# ─── Archivos raw ─────────────────────────────────────────────
FILE_SINIESTROS = DATA_RAW / "BBDD ONSV - SINIESTROS FATALES 2021-2025 (preliminar)(SINIESTROS).csv"
FILE_PERSONAS  = DATA_RAW / "BBDD ONSV - PERSONAS 2021-2025 (preliminar)(PERSONAS INVOLUCRADAS).csv"
FILE_PERU      = DATA_EXTERNAL / "PERU.csv"

# ─── Archivos procesados ──────────────────────────────────────
FILE_SINIESTROS_CLEAN = DATA_PROCESSED / "siniestros_clean.parquet"
FILE_PERSONAS_CLEAN   = DATA_PROCESSED / "personas_clean.parquet"
FILE_MERGED           = DATA_PROCESSED / "dataset_merged.parquet"
FILE_FEATURES         = DATA_PROCESSED / "dataset_features.parquet"
FILE_MODEL_READY      = DATA_PROCESSED / "dataset_model_ready.csv"

# ─── Parámetros de modelo ─────────────────────────────────────
TARGET = "severidad"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Mapeo de severidad basado en cantidad de fallecidos
SEVERITY_MAP = {
    1: 0,  # Baja
    2: 1,  # Media
    3: 2,  # Alta
}

SEVERITY_LABELS = {0: "Baja (1 fallecido)", 1: "Media (2 fallecidos)", 2: "Alta (3+ fallecidos)"}

# ─── Columnas de interés ──────────────────────────────────────
SINIESTROS_KEY_COLS = [
    "CODIGO_SINIESTRO", "FECHA_SINIESTRO", "HORA_SINIESTRO",
    "CLASE_SINIESTRO", "CANTIDAD_FALLECIDOS", "CANTIDAD_LESIONADOS",
    "CANTIDAD_VEHICULOS", "DEPARTAMENTO", "PROVINCIA", "DISTRITO",
    "ZONA", "TIPO_VIA", "RED_VIAL", "COD_CARRETERA",
    "LATITUD", "LONGITUD", "CONDICION_CLIMATICA", "ZONIFICACION",
    "CARACTERISTICAS_VIA", "PERFIL_VIA", "SUPERFICIE_CALZADA",
    "EXISTE_SENIAL_VERTICAL", "CLASIFICACION_SENIAL_VERTICAL_1",
    "CLASIFICACION_SENIAL_VERTICAL_2", "EXISTE_SENIAL_HORIZONTAL",
    "CAUSA_FACTOR_PRINCIPAL", "CAUSA_ESPECIFICA"
]
