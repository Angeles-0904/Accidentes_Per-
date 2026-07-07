# FASE 2: COMPRENSION DE LOS DATOS (Data Understanding)

## 2.1 Descripcion de los Datasets

### Dataset 1: Siniestros Fatales
- **Archivo:** `BBDD ONSV - SINIESTROS FATALES 2021-2025 (preliminar)(SINIESTROS).csv`
- **Registros:** 9,107 siniestros
- **Periodo:** 2021 - 2025 (preliminar)
- **Columnas:** 27 variables
- **Descripcion:** Cada fila representa un siniestro vial fatal ocurrido en el Peru, con detalles sobre ubicacion, fecha, hora, clima, tipo de via, causas y consecuencias.

**Variables principales:**
| Variable | Tipo | Descripcion |
|----------|------|-------------|
| CODIGO SINIESTRO | Texto | Identificador unico del siniestro |
| FECHA SINIESTRO | Fecha | Fecha del evento |
| HORA SINIESTRO | Hora | Hora del evento |
| CLASE SINIESTRO | Categorica | Tipo: CHOQUE, DESPISTE, ATROPELLO, etc. |
| CANTIDAD DE FALLECIDOS | Numerica | Numero de victimas fatales |
| CANTIDAD DE LESIONADOS | Numerica | Numero de lesionados |
| DEPARTAMENTO | Categorica | Departamento de ocurrencia |
| CONDICION CLIMATICA | Categorica | Clima al momento del siniestro |
| CAUSA FACTOR PRINCIPAL | Categorica | Causa principal del siniestro |

### Dataset 2: Personas Involucradas
- **Archivo:** `BBDD ONSV - PERSONAS 2021-2025 (preliminar)(PERSONAS INVOLUCRADAS).csv`
- **Registros:** 25,413 personas
- **Columnas:** 33 variables
- **Descripcion:** Cada fila representa una persona involucrada en un siniestro fatal. Incluye datos demograficos, tipo de participacion, gravedad de lesion, licencia y resultados de dosaje etilico.

**Variables principales:**
| Variable | Tipo | Descripcion |
|----------|------|-------------|
| CODIGO SINIESTRO | Texto | FK al dataset de siniestros |
| TIPO PERSONA | Categorica | CONDUCTOR, PASAJERO, PEATON |
| GRAVEDAD | Categorica | FALLECIDO, LESIONADO, ILESO |
| EDAD | Numerica | Edad de la persona |
| SEXO | Categorica | MASCULINO, FEMENINO |
| POSEE LICENCIA | Binaria | SI/NO |
| DOSAJE ETILICO | Binaria | Resultado positivo/negativo |

### Dataset 3: PERU.csv (Historico)
- **Archivo:** `PERU.csv`
- **Registros:** 24 departamentos + total
- **Descripcion:** Tabla agregada de causas de siniestros de transito por region y ano (2008-2024). Fuente: Anuarios Estadisticos Policiales.

## 2.2 Calidad de los Datos

### Hallazgos principales:

| Aspecto | Siniestros | Personas |
|---------|------------|----------|
| **Registros** | 9,107 | 25,413 |
| **Nulos en senializacion** | ~77-94% | - |
| **Nulos en clima/via** | ~4.6% | - |
| **Nulos en dosaje** | - | ~84% |
| **Nulos en lugar defuncion** | - | ~60% |
| **Nulos en edad** | - | ~10.6% |
| **Nulos en nacionalidad** | - | ~13% |

### Decisiones sobre calidad:
1. **Senializacion (77-94% nulos)**: Excluir del modelado por alta tasa de ausencia
2. **Clima, via, perfil (~4.6% nulos)**: Imputar con moda
3. **Dosaje etilico (~84% nulos)**: Usar indicador binario (positivo = 1, resto = 0)
4. **Edad (~10.6% nulos)**: Imputar con mediana del grupo
5. **Coordenadas**: Algunas con formato europeo (coma decimal) - corregidas

## 2.3 Analisis Exploratorio

### Distribucion de Severidad
- **Severidad Baja (1 fallecido):** 8,177 (89.8%)
- **Severidad Media (2 fallecidos):** 616 (6.8%)
- **Severidad Alta (3+ fallecidos):** 314 (3.4%)

### Distribucion Geografica
- **Top 3 departamentos:** Lima, La Libertad, Cusco
- **Zona:** 51% rural, 49% urbana

### Distribucion Temporal
- Tendencia estable entre 2021-2024
- Meses con mayor incidencia: enero, marzo, diciembre

### Causas Principales
1. Imprudencia del conductor
2. En proceso de investigacion
3. Exceso de velocidad
4. Imprudencia del peaton
5. Ebriedad del conductor

### Clase de Siniestro
- CHOQUE: ~35%
- DESPISTE: ~25%
- ATROPELLO: ~15%
- ATROPELLO FUGA: ~8%

## 2.4 Estadistica Inferencial (R)

### Pruebas Chi-cuadrado realizadas:
1. **ZONA vs SEVERIDAD**: p < 0.05 - Relacion significativa
2. **CLIMA vs SEVERIDAD**: p < 0.05 - Relacion significativa  
3. **CAUSA vs SEVERIDAD**: p < 0.05 - Relacion significativa

### ANOVA:
- **EDAD PROMEDIO vs SEVERIDAD**: Se evaluaron diferencias de edad entre niveles de severidad

## 2.5 Variables Derivadas (Feature Engineering)

| Variable | Descripcion |
|----------|-------------|
| ANIO | Ano del siniestro |
| MES | Mes del siniestro |
| DIA_SEMANA | Dia de la semana (0=Lunes a 6=Domingo) |
| FIN_SEMANA | Indicador de fin de semana |
| FRANJA_HORARIA | Madrugada, Maniana, Tarde, Noche |
| TEMPORADA | Verano, Otonio, Invierno, Primavera |
| MACROREGION | Costa, Sierra, Selva |
| RIESGO_INFRAESTRUCTURA | Puntaje compuesto de via |
| RIESGO_CLIMATICO | Indicador de clima adverso |
| SENIALIZACION_DEFICIENTE | Indicador de falta de seniales |
| ES_NOCHE | Indicador de horario nocturno |
| PCT_ALCOHOL | Proporcion de personas con alcohol positivo |
| PCT_LICENCIA | Proporcion de personas con licencia |

---

*Documento de Data Understanding - Proyecto Final UNA-Puno*
