"""
03_feature_engineering.py
Creacion de variables derivadas para el modelado predictivo.
Fase 3: Data Preparation - Feature Engineering.

Solo se crean variables que pueden construirse a partir
de las columnas reales existentes en los datasets.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
from pathlib import Path

from python.utils.config import FILE_SINIESTROS_CLEAN, FILE_PERSONAS_CLEAN, FILE_FEATURES, FIGURES

print("[1] Cargando datos limpios...")
df_sini = pd.read_parquet(FILE_SINIESTROS_CLEAN)
df_per = pd.read_parquet(FILE_PERSONAS_CLEAN)
print(f"   OK Siniestros: {len(df_sini)}, Personas: {len(df_per)}")

# ─── 1. Variables temporales derivadas ────────────────────────
print("\n[2] Creando variables temporales...")

# Anio, Mes, Dia de la semana (ya existen en personas, crear en siniestros)
df_sini['FECHA_SINIESTRO'] = pd.to_datetime(df_sini['FECHA_SINIESTRO'], errors='coerce')
df_sini['ANIO'] = df_sini['FECHA_SINIESTRO'].dt.year
df_sini['MES'] = df_sini['FECHA_SINIESTRO'].dt.month
df_sini['DIA_SEMANA'] = df_sini['FECHA_SINIESTRO'].dt.dayofweek  # 0=Lunes, 6=Domingo
df_sini['FIN_SEMANA'] = (df_sini['DIA_SEMANA'] >= 5).astype(int)

# Franja horaria (madrugada, maniana, tarde, noche)
bins = [-1, 5, 11, 17, 23]
labels = [0, 1, 2, 3]  # 0=madrugada, 1=maniana, 2=tarde, 3=noche
df_sini['FRANJA_HORARIA'] = pd.cut(
    df_sini['HORA'].fillna(12), bins=bins, labels=labels, right=True
).astype(int)

# Temporada del anio
def get_temporada(mes):
    if mes in [12, 1, 2]:
        return 0  # Verano
    elif mes in [3, 4, 5]:
        return 1  # Otonio
    elif mes in [6, 7, 8]:
        return 2  # Invierno
    else:
        return 3  # Primavera

df_sini['TEMPORADA'] = df_sini['MES'].fillna(1).astype(int).apply(get_temporada)

print("   OK: ANIO, MES, DIA_SEMANA, FIN_SEMANA, FRANJA_HORARIA, TEMPORADA")

# ─── 2. Variables geograficas derivadas ────────────────────────
print("\n[3] Creando variables geograficas...")

# Region natural (aproximada por altitud usando coordenadas - simplificado)
# Clasificacion macro-zona
costa_dptos = ['TUMBES', 'PIURA', 'LAMBAYEQUE', 'LA LIBERTAD', 'ANCASH',
               'LIMA', 'CALLAO', 'ICA', 'AREQUIPA', 'MOQUEGUA', 'TACNA']
sierra_dptos = ['CAJAMARCA', 'HUANUCO', 'PASCO', 'JUNIN', 'HUANCAVELICA',
                'AYACUCHO', 'CUSCO', 'APURIMAC', 'PUNO']
selva_dptos = ['AMAZONAS', 'LORETO', 'SAN MARTIN', 'UCAYALI', 'MADRE DE DIOS']

def get_macroregion(dpto):
    dpto = str(dpto).upper().strip()
    if dpto in costa_dptos:
        return 0  # Costa
    elif dpto in sierra_dptos:
        return 1  # Sierra
    elif dpto in selva_dptos:
        return 2  # Selva
    else:
        return 3  # Desconocido

df_sini['MACROREGION'] = df_sini['DEPARTAMENTO'].apply(get_macroregion)

print("   OK: MACROREGION (Costa=0, Sierra=1, Selva=2)")

# ─── 3. Variables de infraestructura y riesgo ──────────────────
print("\n[4] Creando variables de riesgo...")

# Riesgo de infraestructura (combinacion de tipo via + perfil + superficie)
# Puntaje: mayor = peor condicion
infra_map = {
    'TIPO_VIA': {
        'CARRETERA': 0, 'AVENIDA': 1, 'JIRON': 2, 'CALLE': 2, 'OTRO': 2
    },
    'PERFIL_VIA': {
        'PLANA': 0, 'INCLINADA': 1, 'OTRO': 1
    },
    'SUPERFICIE_CALZADA': {
        'ASFALTADA': 0, 'CONCRETO': 0, 'TROCHA': 2, 'AFIRMADO': 1, 'OTRO': 1
    }
}

for col, mapping in infra_map.items():
    if col in df_sini.columns:
        df_sini[col + '_SCORE'] = df_sini[col].astype(str).str.upper().str.strip().map(mapping).fillna(1)

# Riesgo infraestructura compuesto
risk_cols = [c for c in df_sini.columns if c.endswith('_SCORE')]
if risk_cols:
    df_sini['RIESGO_INFRAESTRUCTURA'] = df_sini[risk_cols].sum(axis=1)
    print(f"   OK: RIESGO_INFRAESTRUCTURA (compuesto de {risk_cols})")

# Riesgo climatico (lluvioso/neblina = mayor riesgo)
def riesgo_climatico(clima):
    clima = str(clima).upper().strip()
    if 'LLUVI' in clima or 'NEBLINA' in clima or 'GRANIZ' in clima:
        return 1
    return 0

df_sini['RIESGO_CLIMATICO'] = df_sini['CONDICION_CLIMATICA'].apply(riesgo_climatico)

# Senializacion deficiente
def senial_deficiente(row):
    prob = 0
    if str(row.get('EXISTE_SENIAL_VERTICAL', '')).upper() == 'NO':
        prob += 1
    if str(row.get('EXISTE_SENIAL_HORIZONTAL', '')).upper() == 'NO':
        prob += 1
    return prob

df_sini['SENIALIZACION_DEFICIENTE'] = df_sini.apply(senial_deficiente, axis=1)

print("   OK: RIESGO_CLIMATICO, SENIALIZACION_DEFICIENTE")

# ─── 4. Variables de personas por siniestro ────────────────────
print("\n[5] Agregando variables de personas...")

# Agregaciones por siniestro (ya se hizo en 01, pero refinamos)
agg = df_per.groupby('CODIGO_SINIESTRO').agg({
    'EDAD': ['mean', 'std'],
    'ALCOHOL_POSITIVO': 'sum',
    'POSEE_LICENCIA': 'sum',
    'CODIGO_PERSONA': 'count',
}).reset_index()

agg.columns = ['CODIGO_SINIESTRO', 'EDAD_PROMEDIO', 'EDAD_STD',
               'TOTAL_ALCOHOL', 'TOTAL_LICENCIA', 'TOTAL_PERSONAS']

# Proporciones
agg['PCT_ALCOHOL'] = (agg['TOTAL_ALCOHOL'] / agg['TOTAL_PERSONAS'].replace(0, 1)).round(3)
agg['PCT_LICENCIA'] = (agg['TOTAL_LICENCIA'] / agg['TOTAL_PERSONAS'].replace(0, 1)).round(3)

# Merge con siniestros
df = df_sini.merge(agg, on='CODIGO_SINIESTRO', how='left')

# Rellenar nulos de agregaciones
for col in ['EDAD_PROMEDIO', 'EDAD_STD', 'TOTAL_ALCOHOL', 'TOTAL_LICENCIA',
            'TOTAL_PERSONAS', 'PCT_ALCOHOL', 'PCT_LICENCIA']:
    df[col] = df[col].fillna(0)

print("   OK: Variables agregadas de personas")

# ─── 5. Variable Noche (indicador) ────────────────────────────
df['ES_NOCHE'] = ((df['FRANJA_HORARIA'] == 3) | (df['FRANJA_HORARIA'] == 0)).astype(int)

# ─── 6. Estadisticas finales ──────────────────────────────────
print("\n[6] Resumen del dataset final:")
print(f"   Registros: {len(df)}")
print(f"   Columnas:  {len(df.columns)}")
print(f"   Target (severidad):\n{df['severidad'].value_counts().sort_index().to_string()}")

# Variables numericas disponibles
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ['CODIGO_SINIESTRO']]
print(f"\n   Variables numericas ({len(num_cols)}): {num_cols[:15]}...")

# ─── 7. Guardar dataset con features ──────────────────────────
print(f"\n[7] Guardando dataset con features...")
df.to_parquet(FILE_FEATURES, index=False)
df.to_csv(Path(FILE_FEATURES).with_suffix('.csv'), index=False, encoding='utf-8-sig')
print(f"   OK: {FILE_FEATURES}")
print("\n[OK] Fase 3 - Feature Engineering completado.")
