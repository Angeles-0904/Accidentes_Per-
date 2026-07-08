# INFORME DE PROYECTO DE ANÁLISIS DE DATOS

---

**Título:** Modelado Predictivo de la Severidad de Siniestros Viales Fatales en el Perú (ONSV 2021–2025)

**Autores:** [Nombres del equipo]

**Institución:** Universidad Nacional del Altiplano – Escuela Profesional de Ingeniería de Sistemas

**Fecha:** Julio 2025

---

## Resumen

El presente proyecto de investigación desarrolla un **sistema analítico y predictivo** para identificar los factores de riesgo asociados a la severidad de los siniestros viales fatales en el Perú, utilizando datos del Observatorio Nacional de Seguridad Vial (ONSV) para el período 2021-2025. La metodología empleada fue CRISP-DM (Cross-Industry Standard Process for Data Mining), implementada en Python para el preprocesamiento, modelado y despliegue, complementada con R para el análisis estadístico inferencial. Se analizaron **9,107 siniestros fatales** y **25,413 personas involucradas**, construyendo **22 variables derivadas** para alimentar 6 algoritmos de clasificación: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM y Naive Bayes. El mejor modelo fue **Logistic Regression** con un **F1 Macro de 0.684** y **ROC-AUC de 0.925**. Se implementó un **Dashboard interactivo en Streamlit** con 10 páginas, filtros dinámicos, mapa coroplético y predicción individual, permitiendo a los tomadores de decisiones explorar los hallazgos y simular escenarios de riesgo.

**Palabras clave:** Siniestros viales, Machine Learning, CRISP-DM, Python, R, Streamlit, ONSV, Perú.

---

## 1. Introducción

Los siniestros de tránsito representan una de las principales causas de muerte en el Perú. Según la Organización Mundial de la Salud, aproximadamente 1.3 millones de personas fallecen cada año en el mundo por accidentes de tránsito. En el contexto peruano, el Observatorio Nacional de Seguridad Vial (ONSV) recopila y sistematiza datos sobre siniestros fatales ocurridos a nivel nacional, con el objetivo de generar evidencia científica para la toma de decisiones en políticas de seguridad vial.

El presente proyecto aborda la problemática desde una perspectiva de Ciencia de Datos, aplicando técnicas de Machine Learning para predecir la severidad de los siniestros viales en función de sus características circunstanciales, geográficas, climatológicas y demográficas. La investigación se enmarca en la metodología CRISP-DM, estándar de la industria para proyectos de minería de datos, y utiliza dos lenguajes de programación complementarios: **Python** para la ingeniería de datos, modelado y dashboard, y **R** para el análisis estadístico descriptivo e inferencial.

---

## 2. Descripción del Problema

### 2.1 Contexto

El Perú registra un promedio de aproximadamente 1,800 siniestros fatales por año, con una tendencia preocupante en regiones como Lima, La Libertad y Cusco. El ONSV, como ente rector en materia de seguridad vial, dispone de bases de datos detalladas sobre cada siniestro fatal ocurrido en el país, incluyendo información sobre circunstancias, causas, personas involucradas y condiciones de infraestructura.

### 2.2 Definición del Problema

**Problema:** ¿Es posible predecir la severidad de un siniestro vial fatal (medida en número de fallecidos) a partir de las características del evento, las condiciones del entorno y el perfil de las personas involucradas?

### 2.3 Preguntas de Investigación

1. ¿Cuáles son los factores de riesgo más influyentes en la severidad de los siniestros viales fatales en el Perú?
2. ¿Qué algoritmo de Machine Learning ofrece el mejor rendimiento predictivo para este problema?
3. ¿Existe una relación estadísticamente significativa entre las variables geográficas, climáticas y de infraestructura con la severidad de los siniestros?
4. ¿Es posible construir un sistema interactivo que permita a los tomadores de decisiones explorar los datos y predecir la severidad de nuevos siniestros?

---

## 3. Objetivos

### 3.1 Objetivo General

Desarrollar un sistema analítico y predictivo capaz de identificar los factores de riesgo asociados a la severidad de los siniestros viales fatales en el Perú, mediante técnicas de Ciencia de Datos y Machine Learning, siguiendo la metodología CRISP-DM.

### 3.2 Objetivos Específicos

1. **Caracterizar** el perfil de los siniestros viales fatales en el Perú (2021-2025) mediante estadística descriptiva y análisis exploratorio de datos (EDA).
2. **Identificar** los factores de riesgo más influyentes en la severidad de los siniestros mediante ingeniería de características.
3. **Construir y comparar** 6 modelos de Machine Learning (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, Naive Bayes) utilizando métricas objetivas.
4. **Seleccionar** el modelo con mejor rendimiento mediante validación cruzada y métricas como Accuracy, Precision, Recall, F1 y ROC-AUC.
5. **Interpretar** el modelo ganador mediante coeficientes y SHAP para extraer recomendaciones accionables.
6. **Validar estadísticamente** las relaciones entre factores de riesgo y severidad mediante pruebas chi-cuadrado y ANOVA en R.
7. **Desplegar** un Dashboard interactivo en Streamlit con 10 páginas, filtros dinámicos y predicción individual.

---

## 4. Hipótesis

### Hipótesis General (HG)

**HG:** Existe una relación significativa entre los factores circunstanciales, geográficos y demográficos de los siniestros viales y la severidad de los mismos, medible mediante técnicas de Machine Learning.

### Hipótesis Específicas

| Código | Hipótesis | Prueba Estadística |
|--------|-----------|-------------------|
| **H₁** | La zona (rural/urbana) tiene relación significativa con la severidad del siniestro | Chi-cuadrado |
| **H₂** | La condición climática al momento del siniestro influye en la severidad | Chi-cuadrado |
| **H₃** | La causa principal del siniestro está asociada a su nivel de severidad | Chi-cuadrado |
| **H₄** | La franja horaria (madrugada, mañana, tarde, noche) influye en la severidad | Chi-cuadrado |
| **H₅** | Existen diferencias significativas en la edad promedio de involucrados según el nivel de severidad | ANOVA |
| **H₆** | El riesgo de infraestructura vial difiere significativamente entre niveles de severidad | ANOVA |
| **H₇** | Es posible predecir la severidad con un ROC-AUC superior a 0.70 usando modelos de clasificación | Evaluación ML |

---

## 5. Variables de Estudio

### 5.1 Variable Dependiente

| Variable | Tipo | Descripción | Codificación |
|----------|------|-------------|--------------|
| **Severidad** | Categórica ordinal | Nivel de severidad del siniestro basado en el número de fallecidos | 0 = Baja (1 fallecido), 1 = Media (2 fallecidos), 2 = Alta (3+ fallecidos) |

### 5.2 Variables Independientes

| Variable | Tipo | Descripción |
|----------|------|-------------|
| Año | Numérica | Año de ocurrencia del siniestro |
| Mes | Numérica | Mes de ocurrencia |
| Día de semana | Numérica | Día de la semana (0=Lunes a 6=Domingo) |
| Fin de semana | Binaria | 1 si ocurrió en fin de semana |
| Hora | Numérica | Hora del día |
| Franja horaria | Categórica | Madrugada, Mañana, Tarde, Noche |
| Temporada | Categórica | Verano, Otoño, Invierno, Primavera |
| Departamento | Categórica | Departamento de ocurrencia |
| Provincia | Categórica | Provincia de ocurrencia |
| Distrito | Categórica | Distrito de ocurrencia |
| Zona | Categórica | Urbana o Rural |
| Macrorregión | Categórica | Costa, Sierra, Selva |
| Tipo de vía | Categórica | Carretera, Avenida, Calle, etc. |
| Condición climática | Categórica | Despejado, Lluvioso, Soleado, Neblina, etc. |
| Clase de siniestro | Categórica | Choque, Despiste, Atropello, etc. |
| Causa principal | Categórica | Imprudencia, Exceso de velocidad, Ebriedad, etc. |
| Cantidad de lesionados | Numérica | Número de personas lesionadas |
| Cantidad de vehículos | Numérica | Número de vehículos involucrados |
| Latitud | Numérica | Coordenada geográfica |
| Longitud | Numérica | Coordenada geográfica |
| Riesgo de infraestructura | Numérica | Puntaje compuesto (tipo de vía + perfil + superficie) |
| Riesgo climático | Binaria | 1 si condición climática adversa |
| Señalización deficiente | Numérica | Indicador de falta de señalización |
| Edad promedio | Numérica | Edad promedio de personas involucradas |
| % Alcohol | Numérica | Proporción de personas con dosaje positivo |
| % Licencia | Numérica | Proporción de personas con licencia de conducir |
| Total personas | Numérica | Número total de personas involucradas |
| Es noche | Binaria | 1 si ocurrió entre 18:00 y 06:00 |

---

## 6. Marco Teórico

### 6.1 Metodología CRISP-DM

CRISP-DM (Cross-Industry Standard Process for Data Mining) es un modelo de proceso estándar para proyectos de minería de datos que consta de seis fases iterativas:

1. **Business Understanding:** Comprensión del problema de negocio
2. **Data Understanding:** Recolección y exploración inicial de datos
3. **Data Preparation:** Limpieza, transformación e ingeniería de características
4. **Modeling:** Aplicación de técnicas de modelado
5. **Evaluation:** Evaluación del modelo contra los objetivos de negocio
6. **Deployment:** Despliegue de los resultados

### 6.2 Algoritmos de Machine Learning

**Regresión Logística:** Modelo lineal utilizado para clasificación binaria y multiclase. Estima la probabilidad de pertenencia a cada clase mediante la función logística. Ventajas: interpretabilidad, bajo costo computacional, buen desempeño en datos linealmente separables.

**Árbol de Decisión:** Modelo no paramétrico que particiona el espacio de características mediante reglas jerárquicas. Ventajas: interpretabilidad visual, manejo de relaciones no lineales. Desventajas: propenso a sobreajuste.

**Random Forest:** Conjunto de árboles de decisión entrenados con bootstrap y selección aleatoria de características. Ventajas: robustez, alto rendimiento, manejo de datos faltantes.

**XGBoost:** Algoritmo de gradient boosting optimizado. Utiliza árboles de decisión secuenciales donde cada árbol corrige los errores del anterior. Ventajas: estado del arte en competencias, regularización incorporada.

**SVM (Support Vector Machine):** Algoritmo que encuentra el hiperplano óptimo de separación entre clases. Ventajas: efectivo en espacios de alta dimensión. Desventajas: escalabilidad limitada.

**Naive Bayes:** Clasificador probabilístico basado en el teorema de Bayes con supuesto de independencia condicional. Ventajas: simplicidad, rápido entrenamiento.

### 6.3 Métricas de Evaluación

- **Accuracy:** Proporción de predicciones correctas
- **Precision:** Proporción de positivos correctos entre los clasificados como positivos
- **Recall:** Proporción de positivos correctos entre los positivos reales
- **F1-Score:** Media armónica de precision y recall
- **ROC-AUC:** Área bajo la curva ROC, mide la capacidad discriminativa del modelo
- **Validación Cruzada (K-Fold):** Técnica que divide los datos en K subconjuntos para evaluar la estabilidad del modelo

### 6.4 Pruebas Estadísticas

**Chi-cuadrado (χ²):** Prueba no paramétrica que evalúa la asociación entre dos variables categóricas. La hipótesis nula establece que las variables son independientes.

**ANOVA (Análisis de Varianza):** Prueba paramétrica que compara las medias de tres o más grupos para determinar si existen diferencias significativas.

---

## 7. Método

### 7.1 Tipo de Investigación

La presente investigación es de tipo **aplicada** con enfoque **cuantitativo** y alcance **predictivo-correlacional**. Se utiliza el método **hipotético-deductivo**, partiendo de hipótesis sobre las relaciones entre variables que son contrastadas mediante pruebas estadísticas y modelos predictivos.

### 7.2 Método y Diseño de Investigación

El estudio sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining), compuesta por seis fases iterativas:

| Fase | Descripción | Herramienta |
|------|-------------|-------------|
| 1. Business Understanding | Definición del problema y objetivos | Documentación |
| 2. Data Understanding | EDA y reporte de calidad | Python + R |
| 3. Data Preparation | Limpieza, integración, feature engineering | Python (pandas) |
| 4. Modeling | Entrenamiento de 6 algoritmos | Python (scikit-learn, xgboost) |
| 5. Evaluation | Comparación de métricas + SHAP | Python |
| 6. Deployment | Dashboard interactivo | Streamlit |

El diseño es **no experimental** (no se manipulan variables) y **transversal** (datos recolectados en un período de 5 años).

### 7.3 Muestra / Base de Datos

Se utilizaron **tres bases de datos** proporcionadas por el Observatorio Nacional de Seguridad Vial (ONSV):

| Base de Datos | Registros | Variables | Período |
|---------------|-----------|-----------|---------|
| Siniestros Fatales | 9,107 siniestros | 27 columnas | 2021-2025 |
| Personas Involucradas | 25,413 personas | 33 columnas | 2021-2025 |
| PERU.csv (Histórico) | 24 departamentos | Múltiples años | 2008-2024 |

**Criterios de inclusión:** Todos los siniestros viales fatales registrados por el ONSV con al menos 1 fallecido en el período 2021-2025.

**Cobertura geográfica:** 24 departamentos del Perú, con representación de las tres macroregiones (Costa, Sierra, Selva).

**Procesamiento:** Los datasets fueron integrados mediante la clave `CÓDIGO SINIESTRO`, relacionando cada siniestro con las personas involucradas en el mismo.

---

## 8. Materiales

| Recurso | Especificación | Propósito |
|---------|---------------|-----------|
| **Python 3.11** | Lenguaje de programación | Preprocesamiento, modelado, dashboard |
| **Pandas / NumPy** | Librerías de manipulación de datos | Limpieza y transformación |
| **Scikit-learn** | Librería de Machine Learning | 6 modelos, métricas, validación |
| **XGBoost** | Librería de gradient boosting | Modelo XGBoost |
| **Matplotlib / Seaborn** | Librerías de visualización | Gráficos EDA |
| **Plotly** | Librería de visualización interactiva | Dashboard |
| **Streamlit** | Framework de dashboards | Interfaz de usuario |
| **Joblib** | Serialización de modelos | Guardar/cargar modelos |
| **R 4.6.1** | Lenguaje de programación estadística | Estadística descriptiva e inferencial |
| **tidyverse / ggplot2** | Librerías de R | Manipulación y visualización |
| **corrplot** | Librería de R | Matriz de correlaciones |
| **caret** | Librería de R | Utilidades de modelado |
| **GeoJSON Perú** | Archivo geográfico | Mapa coroplético |
| **Git + GitHub** | Control de versiones | Repositorio del proyecto |
| **Streamlit Cloud** | Plataforma de hosting | Despliegue del dashboard |

---

## 9. Resultados

### 9.1 Resultados Estadísticos Descriptivos

#### 9.1.1 Distribución de la Variable Objetivo

| Nivel de Severidad | Frecuencia | Porcentaje |
|-------------------|------------|------------|
| Baja (1 fallecido) | 8,177 | **89.8%** |
| Media (2 fallecidos) | 616 | **6.8%** |
| Alta (3+ fallecidos) | 314 | **3.4%** |
| **Total** | **9,107** | **100%** |

Se observa un marcado **desequilibrio de clases**, con predominancia de la severidad baja (89.8%). Esto es esperado en datos reales de siniestros viales, donde la mayoria de eventos fatales involucran una sola victima.

![Distribucion de Severidad](severidad_distribucion.png)

#### 9.1.2 Distribucion Geografica

| Departamento | Siniestros | Porcentaje |
|-------------|------------|------------|
| Lima | 1,917 | 21.0% |
| La Libertad | 850 | 9.3% |
| Cusco | 822 | 9.0% |
| Puno | 748 | 8.2% |
| Arequipa | 698 | 7.7% |
| Otros | 4,072 | 44.7% |

**Zona:** 51% rural, 49% urbana.

![Top 15 Departamentos](departamentos_top.png)

#### 9.1.3 Distribucion Temporal

La tendencia de siniestros se mantuvo estable entre 2021-2024, con los meses de **enero, marzo y diciembre** registrando la mayor incidencia. La **noche** (18:00-24:00) concentra la mayor cantidad de siniestros, seguida de la **maniana** (06:00-12:00).

![Distribucion Temporal](temporal_distribucion.png)

![Franja Horaria](franja_horaria.png)

#### 9.1.4 Causas Principales

| Causa | Porcentaje |
|-------|-----------|
| En proceso de investigación | 50.6% |
| Imprudencia del conductor | 38.4% |
| Imprudencia del peaton | 6.5% |
| Negligencia del conductor | 2.6% |

![Causas Principales](causas_principales.png)

#### 9.1.5 Perfil de Personas

- **Tipo:** Conductores (mayoría), seguido de pasajeros y peatones
- **Sexo:** Predominantemente masculino (~85%)
- **Edad promedio:** 38.6 años
- **Licencia:** ~30% posee licencia de conducir
- **Alcohol positivo:** ~3.5% de los casos

---

### 9.2 Resultados Estadísticos Inferenciales

Se realizaron **6 pruebas estadísticas** en R para validar las hipótesis planteadas. Todas resultaron **significativas** (p < 0.05).

#### 9.2.1 Pruebas Chi-cuadrado (χ²)

| Hipótesis | Variable Independiente | Estadístico χ² | p-valor | Decisión |
|-----------|----------------------|----------------|---------|----------|
| H₁ | ZONA | 355.30 | 1.26e-75 | **H₀ rechazada** ✅ |
| H₂ | CONDICIÓN CLIMÁTICA | 99.32 | 4.99e-04 | **H₀ rechazada** ✅ |
| H₃ | CAUSA FACTOR PRINCIPAL | 135.67 | 4.99e-04 | **H₀ rechazada** ✅ |
| H₄ | FRANJA HORARIA | 26.85 | 4.99e-04 | **H₀ rechazada** ✅ |

**Interpretación:** Existe una relación estadísticamente significativa entre las variables categóricas analizadas y la severidad de los siniestros. La zona (rural/urbana) presenta la asociación más fuerte (χ² = 355.30, p < 0.001), indicando que los siniestros en zonas rurales tienden a ser más severos.

#### 9.2.2 ANOVA (Análisis de Varianza)

| Hipótesis | Variable | Estadístico F | p-valor | Decisión |
|-----------|----------|---------------|---------|----------|
| H₅ | EDAD PROMEDIO | 5.76 | 0.003 | **H₀ rechazada** ✅ |
| H₆ | RIESGO INFRAESTRUCTURA | 3.13 | 0.044 | **H₀ rechazada** ✅ |

**Interpretacion:** La edad promedio de las personas involucradas difiere significativamente entre los niveles de severidad. El analisis post-hoc (Tukey HSD) revelo que los siniestros de severidad baja presentan edades promedio **mayores** que los de severidad media (diferencia media = 1.75 anos, p = 0.018). El riesgo de infraestructura tambien muestra diferencias significativas entre niveles de severidad.

![Boxplot Lesionados (R)](r_boxplot_lesionados.png)

![Severidad por Departamento (R)](r_severidad_departamento.png)

---

### 9.3 Resultados de Implementacion del Software

#### 9.3.1 Pipeline de Datos

Se implementó un pipeline modular en Python compuesto por 5 scripts:

| Script | Función | Archivo |
|--------|---------|---------|
| 01_clean_datasets.py | Limpieza e integración de datos | `python/preprocessing/` |
| 02_eda.py | Análisis exploratorio (10 gráficos) | `python/preprocessing/` |
| 03_feature_engineering.py | Creación de 22 variables derivadas | `python/preprocessing/` |
| 04_modelamiento.py | Entrenamiento y comparación de 6 modelos | `python/models/` |
| 05_shap_interpretability.py | Interpretabilidad con SHAP | `python/evaluation/` |

#### 9.3.2 Comparación de Modelos

| Modelo | Accuracy | Precision | Recall | F1 Macro | ROC-AUC | CV F1 | Tiempo (s) |
|--------|----------|-----------|--------|----------|---------|-------|-----------|
| **Logistic Regression** | **0.849** | **0.650** | **0.783** | **0.684** | **0.925** | **0.626** | 7.95 |
| SVM | 0.820 | 0.603 | 0.727 | 0.633 | 0.916 | 0.571 | 108.76 |
| XGBoost | 0.914 | 0.695 | 0.596 | 0.627 | 0.932 | 0.594 | 6.80 |
| Decision Tree | 0.826 | 0.538 | 0.686 | 0.580 | 0.803 | 0.550 | 0.30 |
| Naive Bayes | 0.877 | 0.501 | 0.475 | 0.454 | 0.803 | 0.423 | 0.06 |
| Random Forest | 0.902 | 0.546 | 0.391 | 0.410 | 0.921 | 0.410 | 6.06 |

**Mejor modelo:** **Logistic Regression** — seleccionado por su **F1 Macro de 0.684** (mejor balance precision-recall) y **ROC-AUC de 0.925** (excelente capacidad discriminativa). Ademas, ofrece alta interpretabilidad gracias a sus coeficientes lineales.

![Comparacion de Modelos](comparacion_modelos.png)

#### 9.3.3 Factores Predictivos Mas Importantes

Basado en los coeficientes del modelo Logistic Regression:

1. **Proporción de personas con licencia** (PCT_LICENCIA)
2. **Proporción de personas con alcohol positivo** (PCT_ALCOHOL)
3. **Edad promedio de involucrados** (EDAD_PROMEDIO)
4. **Hora del día** (HORA)
5. **Riesgo de infraestructura** (RIESGO_INFRAESTRUCTURA)
6. **Cantidad de personas involucradas** (TOTAL_PERSONAS)
7. **Zona rural** (reflejado en RIESGO_INFRAESTRUCTURA)

![Feature Importance Top 15](feature_importance_top15.png)

#### 9.3.4 Dashboard Interactivo (Streamlit)

El dashboard desplegado cuenta con **10 páginas** y las siguientes funcionalidades:

| Página | Funcionalidad |
|--------|---------------|
| Inicio | Presentación del proyecto, metodología CRISP-DM |
| Resumen Nacional | KPIs ejecutivos, distribución de severidad, top departamentos |
| Mapa | Mapa interactivo de puntos por severidad, tabla por departamento |
| Análisis Temporal | Evolución anual, estacionalidad mensual, franja horaria |
| Perfil de Personas | Demografía, tipo de persona, licencia, alcohol |
| Factores de Riesgo | Clima, tipo de vía, infraestructura, validación R |
| Dashboard Ejecutivo | Mapa coroplético, alertas 🔴🟡🟢, ranking de riesgo |
| Análisis Estadístico (R) | Resultados de chi-cuadrado, ANOVA, gráficos R |
| Modelos Predictivos | Comparación de 6 modelos, matrices de confusión |
| Predicción Individual | Formulario interactivo + contribución de variables |
| Conclusiones | Hallazgos, recomendaciones, trabajo futuro |

**Características adicionales:**
- Filtros dinámicos por departamento, año, zona y severidad
- KPIs ejecutivos con tarjetas visuales
- Exportación de datos en CSV desde todas las páginas
- Mapa coroplético de Perú por departamento
- Sistema de alertas basado en índice de riesgo compuesto

---

## 10. Conclusiones

### 10.1 Conclusiones Generales

1. **Se logró desarrollar** un sistema analítico y predictivo completo siguiendo la metodología CRISP-DM, integrando Python y R como herramientas complementarias. El sistema permite predecir la severidad de siniestros viales fatales con un **ROC-AUC de 0.925**, superando ampliamente el criterio de éxito establecido (≥ 0.70).

2. **La severidad baja (1 fallecido)** representa el **89.8%** de los casos, lo que confirma la naturaleza desbalanceada del problema. El modelo Logistic Regression demostró ser el más adecuado para este escenario, logrando el mejor F1 Macro (0.684) al balancear correctamente precisión y recall entre las tres clases.

3. **Los factores de riesgo más influyentes** identificados son: la proporción de personas con licencia, la presencia de alcohol, la edad promedio de los involucrados, la hora del día y el riesgo de infraestructura vial. Estos hallazgos coinciden con la literatura académica sobre siniestros viales.

4. **Las 6 pruebas estadísticas realizadas en R** confirmaron que las variables zona (χ² = 355.30, p < 0.001), condición climática (χ² = 99.32, p < 0.001), causa principal (χ² = 135.67, p < 0.001), franja horaria (χ² = 26.85, p < 0.001), edad promedio (F = 5.76, p = 0.003) y riesgo de infraestructura (F = 3.13, p = 0.044) tienen una **relación estadísticamente significativa** con la severidad de los siniestros.

### 10.2 Recomendaciones

1. **Políticas públicas:** Reforzar los controles de velocidad y alcoholemia en carreteras nacionales, especialmente en zonas rurales y durante la noche. Implementar campañas de concientización focalizadas en los departamentos con mayor incidencia (Lima, La Libertad, Cusco).

2. **Mejoras en la recolección de datos:** Estandarizar la codificación de causas a nivel nacional, reducir la alta tasa de "En proceso de investigación" (50.6%), y mejorar el registro de señalización vial (actualmente 77-94% de nulos).

3. **Investigación futura:** Incorporar datos de flujo vehicular, diseño geométrico de vías, y condiciones meteorológicas en tiempo real para mejorar la precisión predictiva. Explorar técnicas de deep learning con datos de mayor granularidad.

4. **Adopción institucional:** El dashboard desarrollado puede ser adoptado por el ONSV como herramienta de soporte a la decisión para la formulación de políticas de seguridad vial basadas en evidencia.

### 10.3 Limitaciones

- Alta proporción de causas "En proceso de investigación" que limitan el análisis causal.
- Ausencia de datos de exposición (flujo vehicular) que impiden calcular tasas reales de siniestralidad.
- Desbalance natural de clases que requiere técnicas especializadas de entrenamiento.
- Los datos de señalización presentan más del 77% de valores nulos.

---

## 11. Referencias

1. Observatorio Nacional de Seguridad Vial (ONSV). (2025). *Base de datos de siniestros fatales 2021-2025*. Ministerio de Transportes y Comunicaciones, Perú.

2. Shearer, C. (2000). *The CRISP-DM model: The new blueprint for data mining*. Journal of Data Warehousing, 5(4), 13-22.

3. Witten, I. H., Frank, E., Hall, M. A., & Pal, C. J. (2016). *Data Mining: Practical Machine Learning Tools and Techniques* (4th ed.). Morgan Kaufmann.

4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

5. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems, 30.

6. Wickham, H., & Grolemund, G. (2017). *R for Data Science*. O'Reilly Media.

7. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference.

8. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD.

9. Organización Mundial de la Salud. (2023). *Global Status Report on Road Safety 2023*. Ginebra.

10. Policía Nacional del Perú. (2024). *Anuario Estadístico Policial 2024*. Dirección de Estadística.

---

**Anexos disponibles en el repositorio:**
- Código fuente completo (Python + R)
- Dashboard interactivo: [Enlace a Streamlit Cloud]
- Dataset procesado
- Figuras y tablas de resultados
- Reporte estadístico de R
