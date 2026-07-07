"""
02_eda.py
Analisis Exploratorio de Datos (EDA).
Fase 2: Data Understanding - Exploracion y visualizacion.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from python.utils.config import FILE_SINIESTROS_CLEAN, FIGURES, REPORTS

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
FIGURES.mkdir(parents=True, exist_ok=True)

print("[1] Cargando datos limpios...")
df = pd.read_parquet(FILE_SINIESTROS_CLEAN)
print(f"   OK {len(df)} registros, {len(df.columns)} columnas")

# ─── 1. Distribucion de variable objetivo ─────────────────────
print("\n[2] Distribucion de severidad")
print(df['severidad'].value_counts().sort_index())
print(f"  % Baja (1 fallecido):  {df['severidad'].value_counts(normalize=True)[0]:.1%}")
print(f"  % Media (2 fallecidos): {df['severidad'].value_counts(normalize=True)[1]:.1%}")
print(f"  % Alta (3+ fallecidos): {df['severidad'].value_counts(normalize=True)[2]:.1%}")

fig, ax = plt.subplots()
df['severidad'].map({0: 'Baja (1)', 1: 'Media (2)', 2: 'Alta (3+)'}).value_counts().plot(
    kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'], ax=ax
)
ax.set_title('Distribucion de Severidad de Siniestros', fontsize=14, fontweight='bold')
ax.set_xlabel('Severidad')
ax.set_ylabel('Cantidad de Siniestros')
for i, v in enumerate(df['severidad'].value_counts().values):
    ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES / 'severidad_distribucion.png', dpi=150)
plt.close()
print("   Grafico guardado: severidad_distribucion.png")

# ─── 2. Distribucion temporal ─────────────────────────────────
print("\n[3] Analisis temporal")
df['ANIO'] = df['FECHA_SINIESTRO'].dt.year
df['MES'] = df['FECHA_SINIESTRO'].dt.month

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Por anio
anio_counts = df['ANIO'].value_counts().sort_index()
axes[0].bar(anio_counts.index, anio_counts.values, color='#3498db')
axes[0].set_title('Siniestros por Anio', fontweight='bold')
axes[0].set_xlabel('Anio')
axes[0].set_ylabel('Cantidad')
for i, v in enumerate(anio_counts.values):
    axes[0].text(anio_counts.index[i], v + 20, str(v), ha='center', fontsize=9)

# Por mes (agregado)
mes_counts = df['MES'].value_counts().sort_index()
meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
axes[1].bar(range(1,13), [mes_counts.get(m,0) for m in range(1,13)], color='#2ecc71')
axes[1].set_title('Siniestros por Mes (todos los anios)', fontweight='bold')
axes[1].set_xlabel('Mes')
axes[1].set_xticks(range(1,13))
axes[1].set_xticklabels(meses, rotation=45)
axes[1].set_ylabel('Cantidad')

plt.tight_layout()
plt.savefig(FIGURES / 'temporal_distribucion.png', dpi=150)
plt.close()

# ─── 3. Distribucion geografica ────────────────────────────────
print("\n[4] Analisis geografico")
top_dptos = df['DEPARTAMENTO'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_dptos)))
bars = ax.barh(range(len(top_dptos)), top_dptos.values, color=colors[::-1])
ax.set_yticks(range(len(top_dptos)))
ax.set_yticklabels(top_dptos.index)
ax.set_title('Top 15 Departamentos con Mas Siniestros Fatales', fontweight='bold')
ax.set_xlabel('Cantidad de Siniestros')
for i, v in enumerate(top_dptos.values):
    ax.text(v + 20, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / 'departamentos_top.png', dpi=150)
plt.close()
print(f"   Top 3: {', '.join(top_dptos.index[:3])}")

# ─── 4. Clase de siniestro ────────────────────────────────────
print("\n[5] Clase de siniestro")
clase_counts = df['CLASE_SINIESTRO'].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(clase_counts)), clase_counts.values, color='#9b59b6')
ax.set_xticks(range(len(clase_counts)))
ax.set_xticklabels(clase_counts.index, rotation=45, ha='right')
ax.set_title('Distribucion por Clase de Siniestro', fontweight='bold')
ax.set_ylabel('Cantidad')
for i, v in enumerate(clase_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / 'clase_siniestro.png', dpi=150)
plt.close()

# ─── 5. Causas principales ────────────────────────────────────
print("\n[6] Causas principales")
causa_counts = df['CAUSA_FACTOR_PRINCIPAL'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(causa_counts)))
ax.barh(range(len(causa_counts)), causa_counts.values, color=colors[::-1])
ax.set_yticks(range(len(causa_counts)))
ax.set_yticklabels(causa_counts.index)
ax.set_title('Top 10 Causas Principales de Siniestros', fontweight='bold')
ax.set_xlabel('Cantidad')
for i, v in enumerate(causa_counts.values):
    ax.text(v + 20, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / 'causas_principales.png', dpi=150)
plt.close()

# ─── 6. Zona y condiciones ────────────────────────────────────
print("\n[7] Zona y condiciones")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Zona
zona_counts = df['ZONA'].value_counts()
axes[0].pie(zona_counts.values, labels=zona_counts.index, autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c', '#95a5a6'], startangle=90)
axes[0].set_title('Zona', fontweight='bold')

# Clima
clima_counts = df['CONDICION_CLIMATICA'].value_counts().head(5)
axes[1].bar(range(len(clima_counts)), clima_counts.values, color='#1abc9c')
axes[1].set_xticks(range(len(clima_counts)))
axes[1].set_xticklabels(clima_counts.index, rotation=45, ha='right', fontsize=8)
axes[1].set_title('Condicion Climatica', fontweight='bold')

# Tipo de via
via_counts = df['TIPO_VIA'].value_counts().head(6)
axes[2].bar(range(len(via_counts)), via_counts.values, color='#e67e22')
axes[2].set_xticks(range(len(via_counts)))
axes[2].set_xticklabels(via_counts.index, rotation=45, ha='right', fontsize=8)
axes[2].set_title('Tipo de Via', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES / 'zona_clima_via.png', dpi=150)
plt.close()

# ─── 7. Severidad por departamento ────────────────────────────
print("\n[8] Severidad por departamento")
severidad_dpto = df.groupby('DEPARTAMENTO')['severidad'].mean().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(severidad_dpto)))
ax.barh(range(len(severidad_dpto)), severidad_dpto.values, color=colors)
ax.set_yticks(range(len(severidad_dpto)))
ax.set_yticklabels(severidad_dpto.index)
ax.set_title('Severidad Promedio por Departamento', fontweight='bold')
ax.set_xlabel('Severidad Promedio (0=Baja, 1=Media, 2=Alta)')
plt.tight_layout()
plt.savefig(FIGURES / 'severidad_departamento.png', dpi=150)
plt.close()

# ─── 8. Franja horaria ────────────────────────────────────────
print("\n[9] Franja horaria")
bins = [0, 6, 12, 18, 24]
labels = ['Madrugada (0-6)', 'Maniana (6-12)', 'Tarde (12-18)', 'Noche (18-24)']
df['FRANJA_HORARIA'] = pd.cut(df['HORA'], bins=bins, labels=labels, right=False)

franja_counts = df['FRANJA_HORARIA'].value_counts()

fig, ax = plt.subplots()
colors_franja = ['#2c3e50', '#f1c40f', '#e67e22', '#8e44ad']
ax.bar(range(len(franja_counts)), franja_counts.values, color=colors_franja)
ax.set_xticks(range(len(franja_counts)))
ax.set_xticklabels(franja_counts.index, rotation=45)
ax.set_title('Siniestros por Franja Horaria', fontweight='bold')
ax.set_ylabel('Cantidad')
for i, v in enumerate(franja_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / 'franja_horaria.png', dpi=150)
plt.close()

# ─── 9. Matriz de correlacion ─────────────────────────────────
print("\n[10] Matriz de correlacion")
cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
# Excluir columnas con demasiados nulos
cols_num = [c for c in cols_num if df[c].isnull().sum() / len(df) < 0.5]

corr = df[cols_num].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, fmt='.2f',
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlacion - Variables Numericas', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES / 'matriz_correlacion.png', dpi=150)
plt.close()

# ─── 10. Severidad vs variables clave ─────────────────────────
print("\n[11] Severidad vs Variables Clave")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Hora vs severidad
axes[0, 0].scatter(df['HORA'], df['severidad'] + np.random.uniform(-0.1, 0.1, len(df)),
                   alpha=0.3, s=5, c='#3498db')
axes[0, 0].set_title('Hora vs Severidad', fontweight='bold')
axes[0, 0].set_xlabel('Hora del dia')
axes[0, 0].set_ylabel('Severidad')

# Zona vs severidad
df.boxplot(column='severidad', by='ZONA', ax=axes[0, 1])
axes[0, 1].set_title('Zona vs Severidad', fontweight='bold')
axes[0, 1].set_xlabel('Zona')

# Clima vs severidad
clima_top = df['CONDICION_CLIMATICA'].value_counts().head(5).index
df_clima = df[df['CONDICION_CLIMATICA'].isin(clima_top)]
df_clima.boxplot(column='severidad', by='CONDICION_CLIMATICA', ax=axes[0, 2])
axes[0, 2].set_title('Clima vs Severidad', fontweight='bold')
axes[0, 2].set_xlabel('Clima')
axes[0, 2].tick_params(axis='x', rotation=45)

# Causa vs severidad
causa_top = df['CAUSA_FACTOR_PRINCIPAL'].value_counts().head(6).index
df_causa = df[df['CAUSA_FACTOR_PRINCIPAL'].isin(causa_top)]
df_causa.boxplot(column='severidad', by='CAUSA_FACTOR_PRINCIPAL', ax=axes[1, 0])
axes[1, 0].set_title('Causa vs Severidad', fontweight='bold')
axes[1, 0].set_xlabel('Causa')
axes[1, 0].tick_params(axis='x', rotation=45)

# Tipo de via vs severidad
via_top = df['TIPO_VIA'].value_counts().head(6).index
df_via = df[df['TIPO_VIA'].isin(via_top)]
df_via.boxplot(column='severidad', by='TIPO_VIA', ax=axes[1, 1])
axes[1, 1].set_title('Tipo de Via vs Severidad', fontweight='bold')
axes[1, 1].set_xlabel('Tipo de Via')
axes[1, 1].tick_params(axis='x', rotation=45)

# Clase siniestro vs severidad
clase_top = df['CLASE_SINIESTRO'].value_counts().head(6).index
df_clase = df[df['CLASE_SINIESTRO'].isin(clase_top)]
df_clase.boxplot(column='severidad', by='CLASE_SINIESTRO', ax=axes[1, 2])
axes[1, 2].set_title('Clase de Siniestro vs Severidad', fontweight='bold')
axes[1, 2].set_xlabel('Clase')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.suptitle('')
plt.tight_layout()
plt.savefig(FIGURES / 'severidad_vs_variables.png', dpi=150)
plt.close()

print("\n[12] EDA completado - Todos los graficos guardados en:")
print(f"    {FIGURES}")
print("\n[OK] Fase 2 - EDA finalizado.")
