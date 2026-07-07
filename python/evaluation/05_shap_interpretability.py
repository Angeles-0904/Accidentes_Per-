"""
05_shap_interpretability.py
Interpretabilidad del modelo ganador usando SHAP.
Fase 5: Evaluation - Interpretabilidad.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[!] SHAP no instalado. Ejecute: pip install shap")

from python.utils.config import FILE_FEATURES, MODELS_DIR, FIGURES, TABLES, REPORTS, RANDOM_STATE, SEVERITY_LABELS

print("=" * 60)
print("FASE 5: INTERPRETABILIDAD CON SHAP")
print("=" * 60)

# ─── 1. Cargar datos y modelo ────────────────────────────────
print("\n[1] Cargando datos y mejor modelo...")
df = pd.read_parquet(FILE_FEATURES)

# Columnas a excluir (mismas que en modelamiento)
# Cargar scaler para conocer las features exactas usadas en el modelo
scaler_path = MODELS_DIR / 'scaler.pkl'
scaler = joblib.load(scaler_path)
feature_cols = list(scaler.feature_names_in_)

X = df[feature_cols].copy()

# Imputar nulos
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

# Cargar el mejor modelo (Logistic Regression)
modelo_path = MODELS_DIR / 'modelo_logistic_regression.pkl'
scaler_path = MODELS_DIR / 'scaler.pkl'

if not modelo_path.exists():
    print("[!] Modelo no encontrado. Buscando alternativas...")
    modelos_disp = list(MODELS_DIR.glob('modelo_*.pkl'))
    modelos_disp = [m for m in modelos_disp if 'scaler' not in m.name]
    if modelos_disp:
        modelo_path = modelos_disp[0]
        print(f"   Usando: {modelo_path.name}")

model = joblib.load(modelo_path)

# Escalar
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_scaled = X.copy()
X_scaled[num_cols] = scaler.transform(X[num_cols])

print(f"   OK Modelo: {modelo_path.name}")
print(f"   OK Datos: {X_scaled.shape}")

# ─── 2. SHAP Analysis ────────────────────────────────────────
if SHAP_AVAILABLE:
    print("\n[2] Calculando valores SHAP...")
    
    # Para Logistic Regression usamos LinearExplainer
    if 'logistic' in str(modelo_path).lower():
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
    else:
        # Para tree-based usamos TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
    
    # Si es multiclase, shap_values es una lista
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        shap_agg = np.abs(shap_values).mean(axis=0)  # Promedio sobre clases
        if n_classes == 3:
            shap_agg_combined = np.abs(shap_values[1]).mean(axis=0)  # Para clase media
    else:
        shap_agg = np.abs(shap_values).mean(axis=0)
    
    # ─── 3. Feature importance SHAP ──────────────────────────
    print("\n[3] Feature importance global...")
    
    if not isinstance(shap_values, list):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': shap_agg
        }).sort_values('importance', ascending=False)
    else:
        # Para multiclase, usar la suma de valores absolutos
        total_importance = np.zeros(len(feature_cols))
        for sv in shap_values:
            total_importance += np.abs(sv).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': total_importance
        }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(TABLES / 'shap_feature_importance.csv', index=False)
    
    # Grafico
    top_n = 15
    plt.figure(figsize=(10, 7))
    top = feature_importance.head(top_n)
    colors = plt.cm.Purples(np.linspace(0.3, 0.9, top_n))
    plt.barh(range(len(top)), top['importance'].values, color=colors[::-1])
    plt.yticks(range(len(top)), top['feature'].values)
    plt.xlabel('Importancia SHAP (promedio |SHAP value|)')
    plt.title(f'Top {top_n} Variables - Importancia SHAP', fontweight='bold', fontsize=13)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES / 'shap_feature_importance.png', dpi=150)
    plt.close()
    print(f"   Grafico guardado: shap_feature_importance.png")
    
    # ─── 4. SHAP Summary plot ────────────────────────────────
    print("\n[4] Generando SHAP summary plot...")
    plt.figure(figsize=(10, 7))
    
    if not isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_scaled, feature_names=feature_cols,
                          show=False, max_display=15)
    else:
        shap.summary_plot(shap_values[1], X_scaled, feature_names=feature_cols,
                          show=False, max_display=15)
    
    plt.tight_layout()
    plt.savefig(FIGURES / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Grafico guardado: shap_summary_plot.png")
    
    # ─── 5. SHAP Bar plot ────────────────────────────────────
    print("\n[5] Generando SHAP bar plot...")
    plt.figure(figsize=(10, 7))
    
    if not isinstance(shap_values, list):
        shap.plots.bar(explainer(X_scaled)[:100], max_display=15, show=False)
    else:
        shap.plots.bar(explainer(X_scaled)[:100], max_display=15, show=False)
    
    plt.tight_layout()
    plt.savefig(FIGURES / 'shap_bar_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Grafico guardado: shap_bar_plot.png")
    
    print("\n[OK] Analisis SHAP completado.")
else:
    print("\n[!] SHAP no disponible. Generando reporte alternativo...")
    # Feature importance alternativa desde coeficientes
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) > 1:
            coef_mean = np.abs(coef).mean(axis=0)
        else:
            coef_mean = np.abs(coef)
        
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': coef_mean
        }).sort_values('importance', ascending=False)
        
        fi.to_csv(TABLES / 'feature_importance.csv', index=False)
        
        top_n = 15
        plt.figure(figsize=(10, 7))
        top = fi.head(top_n)
        colors = plt.cm.Purples(np.linspace(0.3, 0.9, top_n))
        plt.barh(range(len(top)), top['importance'].values, color=colors[::-1])
        plt.yticks(range(len(top)), top['feature'].values)
        plt.xlabel('|Coeficiente| promedio')
        plt.title(f'Top {top_n} Variables - Coeficientes Logistic Regression', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES / 'feature_importance_top15.png', dpi=150)
        plt.close()
        print(f"   Grafico guardado: feature_importance_top15.png")

print("\n[OK] Fase 5 - Interpretabilidad completada.")
print(f"    Resultados en: {TABLES}")
print(f"    Graficos en:   {FIGURES}")
