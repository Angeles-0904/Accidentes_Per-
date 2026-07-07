"""
04_modelamiento.py
Implementacion y comparacion de 6 algoritmos de clasificacion.
Fase 4: Modeling + Fase 5: Evaluation.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from time import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

from python.utils.config import FILE_FEATURES, MODELS_DIR, FIGURES, TABLES, REPORTS, RANDOM_STATE, TEST_SIZE, CV_FOLDS, SEVERITY_LABELS

sns.set_theme(style="whitegrid")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FASE 4: MODELADO - Comparacion de 6 Algoritmos")
print("=" * 60)

# ─── 1. Cargar datos ──────────────────────────────────────────
print("\n[1] Cargando dataset con features...")
df = pd.read_parquet(FILE_FEATURES)
print(f"   OK {len(df)} registros, {len(df.columns)} columnas")

# ─── 2. Preparar X e y ────────────────────────────────────────
print("\n[2] Preparando variables predictoras y target...")

# Columnas a excluir
excluir = ['CODIGO_SINIESTRO', 'FECHA_SINIESTRO', 'HORA_SINIESTRO',
           'CLASE_SINIESTRO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO',
           'ZONA', 'TIPO_VIA', 'RED_VIAL', 'COD_CARRETERA',
           'CONDICION_CLIMATICA', 'ZONIFICACION', 'CARACTERISTICAS_VIA',
           'PERFIL_VIA', 'SUPERFICIE_CALZADA', 'EXISTE_SENIAL_VERTICAL',
           'CLASIFICACION_SENIAL_VERTICAL_1', 'CLASIFICACION_SENIAL_VERTICAL_2',
           'EXISTE_SENIAL_HORIZONTAL', 'CAUSA_FACTOR_PRINCIPAL', 'CAUSA_ESPECIFICA',
           'CANTIDAD_FALLECIDOS',  # target related
           'FRANJA_HORARIA',  # ya tenemos ES_NOCHE
           'TIPO_VIA_SCORE', 'PERFIL_VIA_SCORE', 'SUPERFICIE_CALZADA_SCORE']

feature_cols = [c for c in df.columns if c not in excluir and c != 'severidad']
print(f"   Predictoras: {len(feature_cols)}")
print(f"   Features: {feature_cols}")

X = df[feature_cols].copy()
y = df['severidad'].copy()

# Manejar nulos
nulos_x = X.isnull().sum().sum()
if nulos_x > 0:
    print(f"   [!] {nulos_x} valores nulos en X, imputando con mediana...")
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

# Distribucion del target
print(f"\n   Distribucion target:")
for k, v in SEVERITY_LABELS.items():
    print(f"      {k} - {v}: {sum(y == k)} ({sum(y == k)/len(y):.1%})")

# ─── 3. Split entrenamiento/prueba ────────────────────────────
print("\n[3] Dividiendo datos (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Escalar variables numericas
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
print("   OK Scaler guardado")

# ─── 4. Definir modelos ───────────────────────────────────────
print("\n[4] Inicializando 6 modelos...")

modelos = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=RANDOM_STATE, max_depth=10, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric='mlogloss', scale_pos_weight=1
    ),
    'SVM': SVC(
        kernel='rbf', probability=True, random_state=RANDOM_STATE, class_weight='balanced'
    ),
    'Naive Bayes': GaussianNB()
}

print(f"   OK {len(modelos)} modelos configurados")

# ─── 5. Entrenar y evaluar ────────────────────────────────────
print("\n" + "=" * 60)
print("ENTRENAMIENTO Y EVALUACION")
print("=" * 60)

resultados = []
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for nombre, model in modelos.items():
    print(f"\n--- {nombre} ---")
    start = time()

    # Entrenar
    if nombre in ['Logistic Regression', 'SVM']:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='f1_macro')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')

    elapsed = time() - start

    # Metricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except:
        roc_auc = 0.0

    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 (macro): {f1:.4f}")
    print(f"   ROC-AUC:    {roc_auc:.4f}")
    print(f"   CV F1:      {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   Tiempo:     {elapsed:.2f}s")

    result = {
        'Modelo': nombre, 'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
        'Recall': round(rec, 4), 'F1': round(f1, 4), 'ROC_AUC': round(roc_auc, 4),
        'CV_F1': round(cv_scores.mean(), 4), 'Tiempo_s': round(elapsed, 2)
    }
    resultados.append(result)

    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Baja', 'Media', 'Alta'],
                yticklabels=['Baja', 'Media', 'Alta'])
    plt.title(f'Matriz de Confusion - {nombre}', fontweight='bold')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(FIGURES / f'cm_{nombre.replace(" ", "_")}.png', dpi=100)
    plt.close()

    # Guardar modelo
    nombre_archivo = nombre.lower().replace(' ', '_')
    joblib.dump(model, MODELS_DIR / f'modelo_{nombre_archivo}.pkl')
    print(f"   Modelo guardado: modelo_{nombre_archivo}.pkl")

# ─── 6. Tabla comparativa ─────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARACION DE MODELOS")
print("=" * 60)

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('F1', ascending=False)

print(df_resultados.to_string(index=False))

# Guardar tabla
df_resultados.to_csv(TABLES / 'comparacion_modelos.csv', index=False)
print(f"\n   Tabla guardada: {TABLES / 'comparacion_modelos.csv'}")

# ─── 7. Grafico comparativo ──────────────────────────────────
print("\n[5] Generando grafico comparativo...")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

metricas = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
x = np.arange(len(metricas))
width = 0.12

for i, row in df_resultados.iterrows():
    valores = [row[m] for m in metricas]
    offset = (i - len(df_resultados)/2 + 0.5) * width
    axes[0].bar(x + offset, valores, width, label=row['Modelo'])

axes[0].set_xticks(x)
axes[0].set_xticklabels(metricas, fontsize=11)
axes[0].set_ylabel('Puntaje')
axes[0].set_title('Comparacion de Metricas por Modelo', fontweight='bold', fontsize=13)
axes[0].legend(loc='lower left', fontsize=9)
axes[0].set_ylim(0, 1.05)
axes[0].grid(axis='y', alpha=0.3)

# CV F1
modelos_nombres = df_resultados['Modelo']
cv_vals = df_resultados['CV_F1']
bars = axes[1].bar(range(len(modelos_nombres)), cv_vals, color=plt.cm.Set2(np.linspace(0, 1, len(modelos_nombres))))
axes[1].set_xticks(range(len(modelos_nombres)))
axes[1].set_xticklabels(modelos_nombres, rotation=30, ha='right')
axes[1].set_ylabel('F1 Macro (CV)')
axes[1].set_title('Validacion Cruzada - F1 Macro Promedio', fontweight='bold', fontsize=13)
axes[1].set_ylim(0, max(cv_vals) * 1.2)
for bar, val in zip(bars, cv_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / 'comparacion_modelos.png', dpi=150)
plt.close()

# ─── 8. Mejor modelo ──────────────────────────────────────────
print("\n[6] Mejor modelo:")
mejor = df_resultados.iloc[0]
print(f"   {mejor['Modelo']} - F1: {mejor['F1']:.4f}, ROC-AUC: {mejor['ROC_AUC']:.4f}")

# Guardar feature importance del mejor
if mejor['Modelo'] in ['Random Forest', 'XGBoost', 'Decision Tree']:
    mejor_idx = df_resultados.index[0]
    if mejor['Modelo'] == 'Random Forest':
        model = joblib.load(MODELS_DIR / 'modelo_random_forest.pkl')
    elif mejor['Modelo'] == 'XGBoost':
        model = joblib.load(MODELS_DIR / 'modelo_xgboost.pkl')
    else:
        model = joblib.load(MODELS_DIR / 'modelo_decision_tree.pkl')

    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importances.to_csv(TABLES / 'feature_importance.csv', index=False)

    # Grafico
    top_n = 15
    plt.figure(figsize=(10, 6))
    top = importances.head(top_n)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))
    plt.barh(range(len(top)), top['importance'].values, color=colors[::-1])
    plt.yticks(range(len(top)), top['feature'].values)
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Variables mas Importantes - {mejor["Modelo"]}', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES / 'feature_importance_top15.png', dpi=150)
    plt.close()
    print(f"   Importancia de features guardada: {TABLES / 'feature_importance.csv'}")

# ─── 9. Metricas detalladas del mejor modelo ──────────────────
print(f"\n[7] Reporte de clasificacion detallado ({mejor['Modelo']}):")
print("-" * 50)

# Reentrenar mejor modelo para reporte detallado
mejor_nombre = mejor['Modelo']
if mejor_nombre in ['Logistic Regression', 'SVM']:
    model_best = joblib.load(MODELS_DIR / f'modelo_{mejor_nombre.lower().replace(" ", "_")}.pkl')
    y_pred_best = model_best.predict(X_test_s)
else:
    model_best = joblib.load(MODELS_DIR / f'modelo_{mejor_nombre.lower().replace(" ", "_")}.pkl')
    y_pred_best = model_best.predict(X_test)

report = classification_report(y_test, y_pred_best,
                                target_names=[SEVERITY_LABELS[0], SEVERITY_LABELS[1], SEVERITY_LABELS[2]])
print(report)

with open(REPORTS / 'classification_report.txt', 'w') as f:
    f.write(f"MEJOR MODELO: {mejor_nombre}\n")
    f.write("=" * 50 + "\n")
    f.write(report)

print(f"\n[OK] Fase 4 y 5 - Modelado y Evaluacion completados.")
print(f"    Resultados en: {TABLES}")
print(f"    Modelos en:    {MODELS_DIR}")
