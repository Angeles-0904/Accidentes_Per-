# 🚦 Modelado Predictivo de la Severidad de Siniestros Viales Fatales en el Perú

**Proyecto Final - Universidad Nacional del Altiplano**
*Escuela Profesional de Ingeniería de Sistemas*

---

## 📋 Descripción del Proyecto

Este proyecto desarrolla un **sistema analítico y predictivo** para identificar los factores de riesgo asociados a la severidad de los siniestros viales fatales en el Perú, utilizando datos del **Observatorio Nacional de Seguridad Vial (ONSV)** para el período 2021-2025.

### Metodología
Proyecto completo siguiendo la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1. Business Understanding | Comprensión del negocio | ✅ |
| 2. Data Understanding | Comprensión de los datos | ✅ |
| 3. Data Preparation | Preparación de datos | ✅ |
| 4. Modeling | Modelado (6 algoritmos) | ✅ |
| 5. Evaluation | Evaluación y comparación | ✅ |
| 6. Deployment | Dashboard interactivo | ✅ |

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Propósito |
|------------|-----------|
| **Python** | Preprocesamiento, feature engineering, ML, evaluación |
| **R** | Estadística descriptiva e inferencial, visualizaciones |
| **Streamlit** | Dashboard interactivo web |
| **Pandas / NumPy** | Manipulación de datos |
| **Scikit-learn** | Modelos de ML (Logistic Regression, DT, RF, SVM, NB) |
| **XGBoost** | Modelo gradient boosting |
| **Matplotlib / Seaborn / Plotly** | Visualizaciones |
| **SHAP** | Interpretabilidad de modelos |

---

## 📁 Estructura del Proyecto

```
Proyecto/
├── data/
│   ├── raw/                              # Datos originales ONSV
│   ├── processed/                        # Datos limpios y transformados
│   └── external/                         # PERU.csv (datos históricos)
├── notebooks/                            # Notebooks de análisis
├── python/
│   ├── preprocessing/
│   │   ├── 01_clean_datasets.py          # Limpieza e integración
│   │   ├── 02_eda.py                     # Análisis exploratorio
│   │   └── 03_feature_engineering.py     # Creación de variables
│   ├── models/
│   │   └── 04_modelamiento.py            # 6 modelos + evaluación
│   ├── evaluation/
│   │   └── 05_shap_interpretability.py   # Interpretabilidad SHAP
│   └── utils/
│       └── config.py                     # Configuración centralizada
├── r/
│   └── estadistica/
│       └── analisis_descriptivo.R        # Estadística descriptiva/inferencial
├── dashboard/
│   └── streamlit/
│       └── app.py                        # Dashboard de 9 páginas
├── outputs/
│   ├── figures/                          # 15+ gráficos generados
│   ├── tables/                           # Tablas de resultados
│   ├── models/                           # 6 modelos entrenados (.pkl)
│   └── reports/                          # Reportes y documentación
└── README.md
```

---

## 🚀 Instalación y Ejecución

### Requisitos

- Python 3.10+
- R 4.x (opcional, para estadística)
- Pip (gestor de paquetes Python)

### Instalación

```bash
# Clonar o navegar al proyecto
cd PROYECTOANALISISDEDATOS

# Instalar dependencias Python
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit joblib pyarrow shap
```

### Ejecución del Pipeline Completo

```bash
# 1. Limpieza e integración de datos
python python/preprocessing/01_clean_datasets.py

# 2. Análisis exploratorio (EDA)
python python/preprocessing/02_eda.py

# 3. Feature engineering
python python/preprocessing/03_feature_engineering.py

# 4. Modelado (6 algoritmos)
python python/models/04_modelamiento.py

# 5. Interpretabilidad SHAP
python python/evaluation/05_shap_interpretability.py

# 6. Dashboard interactivo
streamlit run dashboard/streamlit/app.py
```

### Ejecutar R (opcional)

```r
# En RStudio o R console
source("r/estadistica/analisis_descriptivo.R")
```

---

## 📊 Resultados del Modelado

### Comparación de 6 Algoritmos

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 |
|--------|----------|-----------|--------|-----|---------|-------|
| **Logistic Regression** | **0.849** | **0.650** | **0.783** | **0.684** | **0.925** | **0.626** |
| SVM | 0.820 | 0.603 | 0.727 | 0.633 | 0.916 | 0.571 |
| XGBoost | 0.914 | 0.695 | 0.596 | 0.627 | 0.932 | 0.594 |
| Decision Tree | 0.826 | 0.538 | 0.686 | 0.580 | 0.803 | 0.550 |
| Naive Bayes | 0.877 | 0.501 | 0.475 | 0.454 | 0.803 | 0.423 |
| Random Forest | 0.902 | 0.546 | 0.391 | 0.410 | 0.921 | 0.410 |

### Mejor Modelo: Logistic Regression
- **F1 Macro:** 0.684
- **ROC-AUC:** 0.925
- **Ventajas:** Interpretabilidad, balance precision-recall, bajo costo computacional

### Factores Más Predictivos
1. Proporción de personas con licencia
2. Proporción de personas con alcohol positivo
3. Promedio de edad de involucrados
4. Hora del día
5. Riesgo de infraestructura

---

## 📈 Dashboard Streamlit

El dashboard interactivo incluye **9 páginas**:

| Página | Descripción |
|--------|-------------|
| **Inicio** | Presentación del proyecto y KPIs |
| **Resumen Nacional** | Métricas globales y distribuciones |
| **Mapa** | Distribución geográfica interactiva |
| **Análisis Temporal** | Evolución temporal y estacionalidad |
| **Perfil de Personas** | Demografía de involucrados |
| **Factores de Riesgo** | Causas, clima, vías |
| **Modelos Predictivos** | Comparación de modelos y métricas |
| **Predicción Individual** | Formulario interactivo de predicción |
| **Conclusiones** | Hallazgos clave y recomendaciones |

```bash
# Iniciar dashboard
streamlit run dashboard/streamlit/app.py
```

---

## 🧪 Resultados Clave

1. **89.8%** de siniestros tienen severidad baja (1 fallecido)
2. **Lima** concentra la mayor cantidad de siniestros
3. **Imprudencia del conductor** es la causa principal (~30%)
4. **Zona rural** presenta mayor severidad promedio
5. **Logistic Regression** es el mejor modelo (F1=0.684, ROC-AUC=0.925)

---

## 📚 Referencias

- Observatorio Nacional de Seguridad Vial (ONSV): https://www.onsv.gob.pe/
- Policía Nacional del Perú - UPIAT
- Metodología CRISP-DM
- Universidad Nacional del Altiplano - EPIS

---

## 👥 Autor

**Proyecto Final - Ingeniería de Sistemas**
Universidad Nacional del Altiplano - Puno, Perú
© 2025

---

*"La seguridad vial no es accidental"*
