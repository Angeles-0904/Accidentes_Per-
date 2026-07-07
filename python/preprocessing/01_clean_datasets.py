"""
01_clean_datasets.py
Limpieza y estandarización de los datasets raw de ONSV.
Fase 2: Data Understanding - Preparacion inicial.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from python.utils.config import (
        FILE_SINIESTROS, FILE_PERSONAS,
        FILE_SINIESTROS_CLEAN, FILE_PERSONAS_CLEAN,
        FIGURES, REPORTS
    )
except ImportError:
    from utils.config import (
        FILE_SINIESTROS, FILE_PERSONAS,
        FILE_SINIESTROS_CLEAN, FILE_PERSONAS_CLEAN,
        FIGURES, REPORTS
    )

# ─── 1. Cargar Siniestros ─────────────────────────────────────
print("[1] Cargando siniestros...")
df_sini = pd.read_csv(
    FILE_SINIESTROS, sep=';', encoding='latin1',
    skiprows=4, low_memory=False, header=None
)

# Asignar nombres de columnas manualmente
df_sini.columns = [
    'CODIGO_SINIESTRO', 'FECHA_SINIESTRO', 'HORA_SINIESTRO',
    'CLASE_SINIESTRO', 'CANTIDAD_FALLECIDOS', 'CANTIDAD_LESIONADOS',
    'CANTIDAD_VEHICULOS', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO',
    'ZONA', 'TIPO_VIA', 'RED_VIAL', 'COD_CARRETERA',
    'LATITUD', 'LONGITUD', 'CONDICION_CLIMATICA', 'ZONIFICACION',
    'CARACTERISTICAS_VIA', 'PERFIL_VIA', 'SUPERFICIE_CALZADA',
    'EXISTE_SENIAL_VERTICAL', 'CLASIFICACION_SENIAL_VERTICAL_1',
    'CLASIFICACION_SENIAL_VERTICAL_2', 'EXISTE_SENIAL_HORIZONTAL',
    'CAUSA_FACTOR_PRINCIPAL', 'CAUSA_ESPECIFICA'
]

print(f"   OK {len(df_sini)} registros de siniestros cargados")

# ─── 2. Cargar Personas ───────────────────────────────────────
print("[2] Cargando personas involucradas...")
df_per = pd.read_csv(
    FILE_PERSONAS, sep=';', encoding='latin1',
    skiprows=3, low_memory=False, header=None
)

# Asignar nombres
col_names_per = [
    'CODIGO_SINIESTRO', 'CODIGO_VEHICULO', 'CODIGO_PERSONA',
    'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO',
    'TIPO_PERSONA', 'GRAVEDAD', 'LUGAR_ATENCION_LESIONADO',
    'LUGAR_DEFUNCION', 'SITUACION_PERSONA', 'PAIS_NACIONALIDAD',
    'EDAD', 'SEXO', 'POSEE_LICENCIA',
    'ESTADO_LICENCIA', 'CLASE_LICENCIA',
    'DOSAJE_CUALITATIVO_SOMETIDO',
    'RESULTADO_DOSAJE_CUALITATIVO',
    'DOSAJE_CUANTITATIVO_SOMETIDO',
    'VEHICULO', 'FECHA', 'ANIO', 'MES', 'DIA', 'HORA',
    'CLASE_SINIESTRO_PERSONA', 'CAUSA', 'CAUSA_ESPECIFICA_PERSONA',
    'TIPO_VIA_PERSONA', 'COD_CARRETERA_PERSONA', 'RED_VIAL_PERSONA'
]

for i, name in enumerate(col_names_per):
    df_per.rename(columns={i: name}, inplace=True)

# Descartar columnas extras si las hay
df_per = df_per[col_names_per]

print(f"   OK {len(df_per)} registros de personas cargados")

# ─── 3. Limpieza Siniestros ───────────────────────────────────
print("[3] Limpiando siniestros...")

# Columnas numericas
for col in ['CANTIDAD_FALLECIDOS', 'CANTIDAD_LESIONADOS', 'CANTIDAD_VEHICULOS']:
    df_sini[col] = pd.to_numeric(df_sini[col], errors='coerce').fillna(0).astype(int)

# Coordenadas: limpiar comas (decimales con coma)
for col in ['LATITUD', 'LONGITUD']:
    df_sini[col] = (
        df_sini[col].astype(str).str.replace(',', '.', regex=False).str.strip()
    )
    df_sini[col] = pd.to_numeric(df_sini[col], errors='coerce')

# Fecha
df_sini['FECHA_SINIESTRO'] = pd.to_datetime(
    df_sini['FECHA_SINIESTRO'], dayfirst=True, errors='coerce'
)

# Hora
def parse_hora(h):
    try:
        h = str(h).strip()
        if ':' in h:
            return int(h.split(':')[0])
    except (ValueError, TypeError, AttributeError):
        pass
    return np.nan

df_sini['HORA'] = df_sini['HORA_SINIESTRO'].apply(parse_hora)

# Target: severidad
df_sini['severidad'] = df_sini['CANTIDAD_FALLECIDOS'].map({
    1: 0, 2: 1
}).fillna(2).astype(int)

print("   OK siniestros limpiados")

# ─── 4. Limpieza Personas ─────────────────────────────────────
print("[4] Limpiando personas...")

df_per['EDAD'] = pd.to_numeric(df_per['EDAD'], errors='coerce')

# Estandarizar campos categoricos
for col in ['SEXO', 'GRAVEDAD', 'TIPO_PERSONA']:
    df_per[col] = df_per[col].astype(str).str.strip().str.upper()

# Licencia
df_per['POSEE_LICENCIA'] = (
    df_per['POSEE_LICENCIA'].astype(str).str.strip().str.upper()
)
df_per['POSEE_LICENCIA'] = df_per['POSEE_LICENCIA'].map({'SI': 1}).fillna(0).astype(int)

# Alcohol positivo
df_per['ALCOHOL_POSITIVO'] = (
    df_per['RESULTADO_DOSAJE_CUALITATIVO']
    .astype(str).str.strip().str.upper()
    .map({'POSITIVO': 1}).fillna(0).astype(int)
)

print("   OK personas limpiadas")

# ─── 5. Agregaciones por siniestro ────────────────────────────
print("[5] Agregando datos de personas por siniestro...")

agg = df_per.groupby('CODIGO_SINIESTRO').agg({
    'EDAD': ['mean', 'median', 'min', 'max'],
    'SEXO': lambda x: (x.str.upper() == 'MASCULINO').sum(),
    'TIPO_PERSONA': lambda x: (x.str.upper() == 'CONDUCTOR').sum(),
    'GRAVEDAD': [
        lambda x: (x.str.upper() == 'FALLECIDO').sum(),
        lambda x: (x.str.upper() == 'LESIONADO').sum(),
    ],
    'POSEE_LICENCIA': 'sum',
    'ALCOHOL_POSITIVO': 'sum',
    'CODIGO_PERSONA': 'count',
}).reset_index()

# Aplanar columnas multiindice
agg.columns = [
    'CODIGO_SINIESTRO', 'EDAD_PROMEDIO', 'EDAD_MEDIANA',
    'EDAD_MIN', 'EDAD_MAX',
    'TOTAL_MASCULINOS', 'TOTAL_CONDUCTORES',
    'TOTAL_FALLECIDOS_AGG', 'TOTAL_LESIONADOS_AGG',
    'TOTAL_LICENCIA', 'TOTAL_ALCOHOL_POSITIVO',
    'TOTAL_PERSONAS'
]

print(f"   OK {len(agg)} siniestros con datos agregados")

# ─── 6. Merge final ──────────────────────────────────────────
print("[6] Integrando datasets...")
df_merged = df_sini.merge(agg, on='CODIGO_SINIESTRO', how='left')

print(f"   OK {len(df_merged)} registros en dataset integrado")

# ─── 7. Reporte de calidad ─────────────────────────────────────
print("\n[7] Reporte de Calidad de Datos")
print("=" * 50)

for name, df in [("Siniestros", df_sini), ("Personas", df_per), ("Integrado", df_merged)]:
    print(f"\n--- {name} ---")
    print(f"  Registros: {len(df)}")
    print(f"  Columnas: {len(df.columns)}")
    nulos = df.isnull().sum()
    nulos_gt0 = nulos[nulos > 0].sort_values(ascending=False)
    if len(nulos_gt0) > 0:
        print(f"  Columnas con nulos ({len(nulos_gt0)}):")
        for col, val in nulos_gt0.head(10).items():
            pct = val / len(df) * 100
            print(f"    {col}: {val} ({pct:.1f}%)")

# ─── 8. Guardar ───────────────────────────────────────────────
print("\n[8] Guardando datasets limpios...")
DATA_PROCESSED = Path(FILE_SINIESTROS_CLEAN).parent
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

df_sini.to_parquet(FILE_SINIESTROS_CLEAN, index=False)
df_per.to_parquet(FILE_PERSONAS_CLEAN, index=False)
df_merged.to_parquet(DATA_PROCESSED / "dataset_merged.parquet", index=False)

# Tambien guardar una copia en CSV para R
df_merged.to_csv(DATA_PROCESSED / "dataset_merged.csv", index=False, encoding='utf-8-sig')

print(f"   OK Siniestros limpios: {FILE_SINIESTROS_CLEAN}")
print(f"   OK Personas limpias:   {FILE_PERSONAS_CLEAN}")
print(f"   OK Dataset integrado:  {DATA_PROCESSED / 'dataset_merged.parquet'}")
print("\n[OK] Fase 2 - Limpieza inicial completada.")
