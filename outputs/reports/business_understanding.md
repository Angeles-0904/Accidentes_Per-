# FASE 1: COMPRENSIÓN DEL NEGOCIO (Business Understanding)

## 1.1 Contexto del Problema

Los siniestros de tránsito representan una de las principales causas de muerte en el Perú. El **Observatorio Nacional de Seguridad Vial (ONSV)** recopila y sistematiza datos sobre siniestros fatales ocurridos a nivel nacional, con el objetivo de generar evidencia científica para la toma de decisiones en políticas de seguridad vial.

La base de datos del ONSV contiene información detallada de siniestros fatales ocurridos entre 2021 y 2025 (preliminar), incluyendo:
- Circunstancias del siniestro (fecha, hora, ubicación, clima, tipo de vía)
- Personas involucradas (conductores, pasajeros, peatones)
- Gravedad de las lesiones (fallecido, lesionado, ileso)
- Factores causales (causa principal y específica)
- Condiciones de infraestructura vial

## 1.2 Objetivo General

Desarrollar un **sistema analítico y predictivo** capaz de identificar los factores de riesgo asociados a la severidad de los siniestros viales fatales en el Perú, mediante técnicas de Ciencia de Datos y Machine Learning, siguiendo la metodología CRISP-DM.

## 1.3 Objetivos Específicos

1. **Caracterizar** el perfil de los siniestros viales fatales en el Perú (2021-2025) mediante estadística descriptiva y análisis exploratorio.
2. **Identificar** los factores de riesgo más influyentes en la severidad de los siniestros (número de fallecidos por evento).
3. **Construir y comparar** al menos 6 modelos de Machine Learning para predecir la severidad.
4. **Seleccionar** el modelo con mejor rendimiento mediante métricas objetivas (Accuracy, Precision, Recall, F1, ROC-AUC).
5. **Interpretar** el modelo ganador mediante SHAP para extraer recomendaciones accionables.
6. **Desplegar** un Dashboard interactivo en Streamlit para visualización y predicción en tiempo real.

## 1.4 Definición del Problema de Machine Learning

| Aspecto | Definición |
|---------|-----------|
| **Tipo de problema** | Clasificación supervisada (multiclase) |
| **Variable objetivo** | `severidad` → 0: Baja (1 fallecido), 1: Media (2 fallecidos), 2: Alta (3+ fallecidos) |
| **Unidad de análisis** | Siniestro (registro individual) |
| **Variables predictoras** | Circunstancias del siniestro + características agregadas de personas involucradas |
| **Métrica principal** | ROC-AUC (macro promedio) |
| **Métricas secundarias** | Accuracy, Precision, Recall, F1, Matriz de Confusión |

### Justificación de la variable objetivo

La base de datos contiene exclusivamente siniestros fatales (al menos 1 fallecido). Por tanto, la severidad se define en función del número de víctimas fatales:

- **Severidad Baja (0):** 1 fallecido (mayoría de casos)
- **Severidad Media (1):** 2 fallecidos
- **Severidad Alta (2):** 3 o más fallecidos

Esta discretización permite transformar un problema de regresión (con pocos valores enteros) en uno de clasificación con 3 niveles interpretables.

## 1.5 Criterios de Éxito

- **ROC-AUC macro promedio** ≥ 0.70 en el conjunto de prueba
- Dashboard funcional con 9 páginas interactivas
- Documentación completa siguiendo CRISP-DM
- Código modular, reutilizable y documentado (PEP8)

## 1.6 Stakeholders

| Stakeholder | Interés |
|-------------|---------|
| **ONSV** | Generar evidencia para políticas de seguridad vial |
| **PNP (UPIAT)** | Mejorar la investigación de siniestros |
| **Ministerio de Transportes** | Identificar tramos críticos y causas frecuentes |
| **Ministerio de Salud** | Reducir mortalidad por siniestros viales |
| **Comunidad académica (UNA-Puno)** | Aportar conocimiento científico |

## 1.7 Recursos Disponibles

- 3 bases de datos CSV (Siniestros Fatales, Personas Involucradas, PERU histórico)
- Python con librerías de ciencia de datos
- R con librerías estadísticas
- Streamlit para dashboard
- Metodología CRISP-DM

## 1.8 Plan de Trabajo CRISP-DM

| Fase | Duración estimada | Entregable |
|------|-------------------|------------|
| 1. Business Understanding | Completado | ✅ Este documento |
| 2. Data Understanding | Siguiente | Reporte EDA + Estadísticas R |
| 3. Data Preparation | Posterior | Dataset limpio + features |
| 4. Modeling | Posterior | 6 modelos entrenados |
| 5. Evaluation | Posterior | Comparación + SHAP |
| 6. Deployment | Posterior | Dashboard Streamlit |

---

*Documento elaborado para el Proyecto Final - Universidad Nacional del Altiplano*
*Escuela Profesional de Ingeniería de Sistemas*
