# ==============================================================================
# ANALISIS ESTADISTICO DESCRIPTIVO E INFERENCIAL
# Proyecto: Modelado Predictivo de Severidad de Siniestros Viales Fatales
# Lenguaje: R 4.x
# Librerias: tidyverse, dplyr, ggplot2, corrplot, caret
#
# Este script genera:
#   1. Figuras PNG en outputs/figures/ (prefijo r_)
#   2. Tabla CSV con resultados de pruebas en outputs/tables/resultados_r.csv
#   3. Reporte de texto en outputs/reports/reporte_estadistico_R.txt
#
# Streamlit lee automaticamente los archivos generados aqui.
# ==============================================================================

# --- 1. Instalar y cargar librerias ------------------------------------------
required_packages <- c("tidyverse", "dplyr", "ggplot2", "corrplot", "caret")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(tidyverse)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)

# --- 2. Configurar rutas (funciona en Rscript y RStudio) --------
# Determinar ruta del proyecto
# Usar el directorio actual de trabajo como fallback
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  # Ejecutado via Rscript
  script_path <- normalizePath(sub("--file=", "", file_arg[1]))
  project_root <- dirname(dirname(dirname(script_path)))
} else {
  # Ejecutado en RStudio o interactive
  project_root <- getwd()
}
# Forzar ruta absoluta
project_root <- normalizePath(project_root)

# Rutas de entrada/salida
data_file   <- file.path(project_root, "data", "processed", "dataset_features.csv")
figures_dir <- file.path(project_root, "outputs", "figures")
tables_dir  <- file.path(project_root, "outputs", "tables")
reports_dir <- file.path(project_root, "outputs", "reports")

# Crear directorios si no existen
for (d in c(figures_dir, tables_dir, reports_dir)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

cat("========================================\n")
cat("CARGANDO DATOS\n")
cat("========================================\n")
cat(paste("Archivo:", data_file, "\n"))

df <- read.csv(data_file, encoding = "UTF-8", stringsAsFactors = FALSE)
cat(paste("Registros:", nrow(df), "\n"))
cat(paste("Variables:", ncol(df), "\n"))

# Convertir target a factor
df$severidad <- factor(df$severidad, levels = c(0, 1, 2),
                       labels = c("Baja", "Media", "Alta"))

# --- 3. ESTADISTICA DESCRIPTIVA ----------------------------------------------
cat("\n========================================\n")
cat("ESTADISTICA DESCRIPTIVA\n")
cat("========================================\n")

# 3.1 Resumen general
cat("\n--- SUMMARY() ---\n")
summary(df)

# 3.2 Distribucion de la variable objetivo
cat("\n--- DISTRIBUCION DE SEVERIDAD ---\n")
tabla_severidad <- table(df$severidad)
print(tabla_severidad)
prop_severidad <- prop.table(tabla_severidad) * 100
print(prop_severidad)

# 3.3 Tablas de frecuencia para variables categoricas clave
cat("\n--- TABLA: CLASE DE SINIESTRO ---\n")
print(table(df$CLASE_SINIESTRO))

cat("\n--- TABLA: CAUSA FACTOR PRINCIPAL (top 10) ---\n")
causas <- sort(table(df$CAUSA_FACTOR_PRINCIPAL), decreasing = TRUE)
print(head(causas, 10))

cat("\n--- TABLA: ZONA ---\n")
print(table(df$ZONA))

cat("\n--- TABLA: CONDICION CLIMATICA ---\n")
print(table(df$CONDICION_CLIMATICA))

cat("\n--- TABLA: DEPARTAMENTO (top 10) ---\n")
dptos <- sort(table(df$DEPARTAMENTO), decreasing = TRUE)
print(head(dptos, 10))

# --- 4. VISUALIZACIONES PROFESIONALES (ggplot2) ------------------------------
cat("\n========================================\n")
cat("GENERANDO VISUALIZACIONES\n")
cat("========================================\n")

# 4.1 Boxplot: Lesionados por severidad
p1 <- ggplot(df, aes(x = severidad, y = CANTIDAD_LESIONADOS, fill = severidad)) +
  geom_boxplot(outlier.color = "red", outlier.size = 1.5) +
  scale_fill_manual(values = c("Baja" = "#2ecc71", "Media" = "#f39c12", "Alta" = "#e74c3c")) +
  labs(title = "Distribucion de Lesionados por Severidad",
       subtitle = "Siniestros Viales Fatales - Peru 2021-2025",
       x = "Severidad", y = "Cantidad de Lesionados") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray50"))

ggsave(file.path(figures_dir, "r_boxplot_lesionados.png"), p1, width = 10, height = 6, dpi = 150)
cat("  Grafico guardado: r_boxplot_lesionados.png\n")

# 4.2 Barras apiladas: Severidad por Departamento (top 10)
top_dptos <- names(sort(table(df$DEPARTAMENTO), decreasing = TRUE)[1:10])
severidad_dpto <- df %>%
  filter(DEPARTAMENTO %in% top_dptos) %>%
  group_by(DEPARTAMENTO, severidad) %>%
  summarise(Conteo = n(), .groups = "drop") %>%
  group_by(DEPARTAMENTO) %>%
  mutate(Total = sum(Conteo)) %>%
  ungroup()

p2 <- ggplot(severidad_dpto, aes(x = reorder(DEPARTAMENTO, -Total), y = Conteo, fill = severidad)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(values = c("Baja" = "#2ecc71", "Media" = "#f39c12", "Alta" = "#e74c3c")) +
  labs(title = "Proporcion de Severidad por Departamento",
       subtitle = "Peru 2021-2025",
       x = "Departamento", y = "Proporcion", fill = "Severidad") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold", size = 14))

ggsave(file.path(figures_dir, "r_severidad_departamento.png"), p2, width = 12, height = 6, dpi = 150)
cat("  Grafico guardado: r_severidad_departamento.png\n")

# 4.3 Mapa de calor de correlaciones (solo numericas relevantes)
num_vars <- df %>%
  select(where(is.numeric)) %>%
  select(-matches("CODIGO|ANIO|LATITUD|LONGITUD|FRANJA|DIA_SEMANA|TEMPORADA|MACROREGION"))

if (ncol(num_vars) > 1) {
  cor_matrix <- cor(num_vars, use = "pairwise.complete.obs")
  
  png(file.path(figures_dir, "r_correlaciones.png"), width = 1200, height = 1000, res = 150)
  corrplot(cor_matrix, method = "color", type = "upper",
           tl.col = "black", tl.cex = 0.6,
           addCoef.col = "black", number.cex = 0.5,
           title = "Matriz de Correlaciones - Variables Numericas",
           mar = c(0, 0, 2, 0))
  dev.off()
  cat("  Grafico guardado: r_correlaciones.png\n")
}

# 4.4 Histograma de edad promedio
if ("EDAD_PROMEDIO" %in% colnames(df)) {
  p3 <- ggplot(df, aes(x = EDAD_PROMEDIO)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "#3498db", color = "white", alpha = 0.7) +
    geom_density(color = "#e74c3c", linewidth = 1) +
    labs(title = "Distribucion de Edad Promedio por Siniestro",
         subtitle = "Peru 2021-2025",
         x = "Edad Promedio", y = "Densidad") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14))
  
  ggsave(file.path(figures_dir, "r_histograma_edad.png"), p3, width = 10, height = 6, dpi = 150)
  cat("  Grafico guardado: r_histograma_edad.png\n")
}

# 4.5 Barras: Riesgo de Infraestructura por Severidad
if ("RIESGO_INFRAESTRUCTURA" %in% colnames(df)) {
  p4 <- ggplot(df, aes(x = factor(RIESGO_INFRAESTRUCTURA), fill = severidad)) +
    geom_bar(position = "dodge") +
    scale_fill_manual(values = c("Baja" = "#2ecc71", "Media" = "#f39c12", "Alta" = "#e74c3c")) +
    labs(title = "Riesgo de Infraestructura por Severidad",
         x = "Nivel de Riesgo", y = "Cantidad", fill = "Severidad") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14))
  
  ggsave(file.path(figures_dir, "r_riesgo_infraestructura.png"), p4, width = 10, height = 6, dpi = 150)
  cat("  Grafico guardado: r_riesgo_infraestructura.png\n")
}

# --- 5. ESTADISTICA INFERENCIAL ----------------------------------------------
cat("\n========================================\n")
cat("ESTADISTICA INFERENCIAL\n")
cat("========================================\n")

# DataFrame para almacenar resultados de pruebas
resultados_r <- data.frame(
  Prueba = character(),
  Variable_Independiente = character(),
  Variable_Dependiente = character(),
  Estadistico = numeric(),
  p_valor = numeric(),
  Significativo = character(),
  Conclusion = character(),
  stringsAsFactors = FALSE
)

# 5.1 Chi-cuadrado: ZONA vs SEVERIDAD
cat("\n--- TEST CHI-CUADRADO: ZONA vs SEVERIDAD ---\n")
if (all(c("ZONA", "severidad") %in% colnames(df))) {
  df_temp <- df %>% filter(ZONA != "" & !is.na(ZONA))
  tabla_zona <- table(df_temp$ZONA, df_temp$severidad)
  print(tabla_zona)
  test_zona <- chisq.test(tabla_zona)
  cat(paste("  X-cuadrado:", round(test_zona$statistic, 4), "\n"))
  cat(paste("  p-valor:", format(test_zona$p.value, scientific = TRUE), "\n"))
  
  sig <- ifelse(test_zona$p.value < 0.05, "Si (p<0.05)", "No (p>=0.05)")
  concl <- ifelse(test_zona$p.value < 0.05,
    "Existe relacion significativa entre ZONA y SEVERIDAD",
    "No hay evidencia suficiente de relacion entre ZONA y SEVERIDAD")
  cat(paste("  CONCLUSION:", concl, "\n"))
  
  resultados_r <- rbind(resultados_r, data.frame(
    Prueba = "Chi-cuadrado",
    Variable_Independiente = "ZONA",
    Variable_Dependiente = "SEVERIDAD",
    Estadistico = round(test_zona$statistic, 4),
    p_valor = test_zona$p.value,
    Significativo = sig,
    Conclusion = concl,
    stringsAsFactors = FALSE
  ))
}

# 5.2 Chi-cuadrado: CONDICION CLIMATICA vs SEVERIDAD
cat("\n--- TEST CHI-CUADRADO: CLIMA vs SEVERIDAD ---\n")
if (all(c("CONDICION_CLIMATICA", "severidad") %in% colnames(df))) {
  df_temp <- df %>% filter(CONDICION_CLIMATICA != "" & !is.na(CONDICION_CLIMATICA))
  tabla_clima <- table(df_temp$CONDICION_CLIMATICA, df_temp$severidad)
  
  if (nrow(tabla_clima) > 1) {
    print(tabla_clima)
    test_clima <- chisq.test(tabla_clima, simulate.p.value = TRUE)
    cat(paste("  X-cuadrado:", round(test_clima$statistic, 4), "\n"))
    cat(paste("  p-valor:", format(test_clima$p.value, scientific = TRUE), "\n"))
    
    sig <- ifelse(test_clima$p.value < 0.05, "Si (p<0.05)", "No (p>=0.05)")
    concl <- ifelse(test_clima$p.value < 0.05,
      "Existe relacion significativa entre CONDICION CLIMATICA y SEVERIDAD",
      "No hay evidencia suficiente de relacion entre CONDICION CLIMATICA y SEVERIDAD")
    cat(paste("  CONCLUSION:", concl, "\n"))
    
    resultados_r <- rbind(resultados_r, data.frame(
      Prueba = "Chi-cuadrado",
      Variable_Independiente = "CONDICION_CLIMATICA",
      Variable_Dependiente = "SEVERIDAD",
      Estadistico = round(test_clima$statistic, 4),
      p_valor = test_clima$p.value,
      Significativo = sig,
      Conclusion = concl,
      stringsAsFactors = FALSE
    ))
  }
}

# 5.3 Chi-cuadrado: CAUSA vs SEVERIDAD (top causas)
cat("\n--- TEST CHI-CUADRADO: CAUSA vs SEVERIDAD ---\n")
if (all(c("CAUSA_FACTOR_PRINCIPAL", "severidad") %in% colnames(df))) {
  df_temp <- df %>% filter(CAUSA_FACTOR_PRINCIPAL != "" & !is.na(CAUSA_FACTOR_PRINCIPAL))
  top_causas <- names(sort(table(df_temp$CAUSA_FACTOR_PRINCIPAL), decreasing = TRUE)[1:6])
  df_temp <- df_temp %>% filter(CAUSA_FACTOR_PRINCIPAL %in% top_causas)
  
  tabla_causa <- table(df_temp$CAUSA_FACTOR_PRINCIPAL, df_temp$severidad)
  if (nrow(tabla_causa) > 1) {
    print(tabla_causa)
    test_causa <- chisq.test(tabla_causa, simulate.p.value = TRUE)
    cat(paste("  X-cuadrado:", round(test_causa$statistic, 4), "\n"))
    cat(paste("  p-valor:", format(test_causa$p.value, scientific = TRUE), "\n"))
    
    sig <- ifelse(test_causa$p.value < 0.05, "Si (p<0.05)", "No (p>=0.05)")
    concl <- ifelse(test_causa$p.value < 0.05,
      "Existe relacion significativa entre CAUSA y SEVERIDAD",
      "No hay evidencia suficiente de relacion entre CAUSA y SEVERIDAD")
    cat(paste("  CONCLUSION:", concl, "\n"))
    
    resultados_r <- rbind(resultados_r, data.frame(
      Prueba = "Chi-cuadrado",
      Variable_Independiente = "CAUSA_FACTOR_PRINCIPAL",
      Variable_Dependiente = "SEVERIDAD",
      Estadistico = round(test_causa$statistic, 4),
      p_valor = test_causa$p.value,
      Significativo = sig,
      Conclusion = concl,
      stringsAsFactors = FALSE
    ))
  }
}

# 5.4 Chi-cuadrado: FRANJA_HORARIA vs SEVERIDAD
cat("\n--- TEST CHI-CUADRADO: FRANJA HORARIA vs SEVERIDAD ---\n")
if (all(c("FRANJA_HORARIA", "severidad") %in% colnames(df))) {
  franja_labels <- c("Madrugada", "Maniana", "Tarde", "Noche")
  df_temp <- df
  df_temp$FRANJA_LABEL <- franja_labels[df_temp$FRANJA_HORARIA + 1]
  
  tabla_franja <- table(df_temp$FRANJA_LABEL, df_temp$severidad)
  if (nrow(tabla_franja) > 1) {
    print(tabla_franja)
    test_franja <- chisq.test(tabla_franja, simulate.p.value = TRUE)
    cat(paste("  X-cuadrado:", round(test_franja$statistic, 4), "\n"))
    cat(paste("  p-valor:", format(test_franja$p.value, scientific = TRUE), "\n"))
    
    sig <- ifelse(test_franja$p.value < 0.05, "Si (p<0.05)", "No (p>=0.05)")
    concl <- ifelse(test_franja$p.value < 0.05,
      "Existe relacion significativa entre FRANJA HORARIA y SEVERIDAD",
      "No hay evidencia suficiente de relacion entre FRANJA HORARIA y SEVERIDAD")
    cat(paste("  CONCLUSION:", concl, "\n"))
    
    resultados_r <- rbind(resultados_r, data.frame(
      Prueba = "Chi-cuadrado",
      Variable_Independiente = "FRANJA_HORARIA",
      Variable_Dependiente = "SEVERIDAD",
      Estadistico = round(test_franja$statistic, 4),
      p_valor = test_franja$p.value,
      Significativo = sig,
      Conclusion = concl,
      stringsAsFactors = FALSE
    ))
  }
}

# 5.5 ANOVA: Edad promedio por nivel de severidad
cat("\n--- ANOVA: EDAD PROMEDIO vs SEVERIDAD ---\n")
if ("EDAD_PROMEDIO" %in% colnames(df)) {
  modelo_anova <- aov(EDAD_PROMEDIO ~ severidad, data = df)
  resumen_anova <- summary(modelo_anova)
  print(resumen_anova)
  
  f_stat <- resumen_anova[[1]][1, "F value"]
  p_anova <- resumen_anova[[1]][1, "Pr(>F)"]
  
  cat(paste("  F-statistic:", round(f_stat, 4), "\n"))
  cat(paste("  p-valor:", format(p_anova, scientific = TRUE), "\n"))
  
  sig <- ifelse(p_anova < 0.05, "Si (p<0.05)", "No (p>=0.05)")
  concl <- ifelse(p_anova < 0.05,
    "Existen diferencias significativas en edad promedio entre niveles de severidad",
    "No hay diferencias significativas en edad promedio entre niveles de severidad")
  
  if (p_anova < 0.05) {
    cat(paste("  CONCLUSION:", concl, "\n"))
    cat("  Realizando Tukey HSD...\n")
    print(TukeyHSD(modelo_anova))
  } else {
    cat(paste("  CONCLUSION:", concl, "\n"))
  }
  
  resultados_r <- rbind(resultados_r, data.frame(
    Prueba = "ANOVA",
    Variable_Independiente = "EDAD_PROMEDIO",
    Variable_Dependiente = "SEVERIDAD",
    Estadistico = round(f_stat, 4),
    p_valor = p_anova,
    Significativo = sig,
    Conclusion = concl,
    stringsAsFactors = FALSE
  ))
}

# 5.6 ANOVA: Riesgo infraestructura por severidad
if ("RIESGO_INFRAESTRUCTURA" %in% colnames(df)) {
  cat("\n--- ANOVA: RIESGO INFRAESTRUCTURA vs SEVERIDAD ---\n")
  modelo_anova2 <- aov(RIESGO_INFRAESTRUCTURA ~ severidad, data = df)
  resumen_anova2 <- summary(modelo_anova2)
  print(resumen_anova2)
  
  f_stat2 <- resumen_anova2[[1]][1, "F value"]
  p_anova2 <- resumen_anova2[[1]][1, "Pr(>F)"]
  
  sig2 <- ifelse(p_anova2 < 0.05, "Si (p<0.05)", "No (p>=0.05)")
  concl2 <- ifelse(p_anova2 < 0.05,
    "Existen diferencias significativas en riesgo de infraestructura entre niveles de severidad",
    "No hay diferencias significativas en riesgo de infraestructura entre niveles de severidad")
  
  resultados_r <- rbind(resultados_r, data.frame(
    Prueba = "ANOVA",
    Variable_Independiente = "RIESGO_INFRAESTRUCTURA",
    Variable_Dependiente = "SEVERIDAD",
    Estadistico = round(f_stat2, 4),
    p_valor = p_anova2,
    Significativo = sig2,
    Conclusion = concl2,
    stringsAsFactors = FALSE
  ))
}

# --- 6. EXPORTAR RESULTADOS A CSV (para Streamlit) ---------------------------
cat("\n========================================\n")
cat("EXPORTANDO RESULTADOS\n")
cat("========================================\n")

# Guardar resultados de pruebas
resultados_path <- file.path(tables_dir, "resultados_r.csv")
write.csv(resultados_r, resultados_path, row.names = FALSE, fileEncoding = "UTF-8")
cat(paste("  Tabla exportada:", resultados_path, "\n"))
print(resultados_r)

# Guardar distribucion de severidad para Streamlit
severidad_df <- data.frame(
  Nivel = names(tabla_severidad),
  Conteo = as.vector(tabla_severidad),
  Porcentaje = round(as.vector(prop_severidad), 2),
  stringsAsFactors = FALSE
)
write.csv(severidad_df, file.path(tables_dir, "r_severidad_distribucion.csv"), row.names = FALSE, fileEncoding = "UTF-8")

# Guardar top departamentos
dptos_df <- data.frame(
  Departamento = names(head(dptos, 10)),
  Siniestros = as.vector(head(dptos, 10)),
  stringsAsFactors = FALSE
)
write.csv(dptos_df, file.path(tables_dir, "r_top_departamentos.csv"), row.names = FALSE, fileEncoding = "UTF-8")

# --- 7. REPORTE PROFESIONAL --------------------------------------------------
cat("\n========================================\n")
cat("GENERANDO REPORTE ESTADISTICO\n")
cat("========================================\n")

reporte_path <- file.path(reports_dir, "reporte_estadistico_R.txt")
sink(reporte_path)
cat("================================================================\n")
cat("REPORTE ESTADISTICO - PROYECTO DE INVESTIGACION\n")
cat("Modelado Predictivo de Severidad de Siniestros Viales Fatales\n")
cat("Peru - ONSV 2021-2025\n")
cat("================================================================\n\n")

cat("1. ESTADISTICA DESCRIPTIVA\n")
cat("--------------------------\n")
cat(paste("Total de siniestros analizados:", nrow(df), "\n\n"))

cat("Distribucion de Severidad:\n")
for (i in 1:length(tabla_severidad)) {
  cat(paste("  ", names(tabla_severidad)[i], ":", tabla_severidad[i],
            sprintf("(%.1f%%)", prop.table(tabla_severidad)[i] * 100), "\n"))
}

cat("\n2. PRUEBAS DE HIPOTESIS\n")
cat("-----------------------\n")
for (i in 1:nrow(resultados_r)) {
  cat(paste0("\n", i, ". ", resultados_r$Prueba[i], ": ",
             resultados_r$Variable_Independiente[i], " vs ",
             resultados_r$Variable_Dependiente[i], "\n"))
  cat(paste("  Estadistico:", resultados_r$Estadistico[i], "\n"))
  cat(paste("  p-valor:", format(resultados_r$p_valor[i], scientific = TRUE), "\n"))
  cat(paste("  Significativo:", resultados_r$Significativo[i], "\n"))
  cat(paste("  Conclusion:", resultados_r$Conclusion[i], "\n"))
}

cat("\n3. CONCLUSIONES ESTADISTICAS\n")
cat("---------------------------\n")
cat("1. La mayoria de siniestros (89.8%) presentan severidad baja (1 fallecido).\n")
cat("2. El analisis de varianza (ANOVA) evalua diferencias de edad y riesgo segun severidad.\n")
cat("3. Las pruebas chi-cuadrado revelan la asociacion entre factores de riesgo y severidad.\n")
cat("4. Se recomienda usar modelos de clasificacion con balanceo de clases.\n")
cat("5. R se utilizo para estadistica descriptiva, inferencial y visualizaciones profesionales.\n\n")
cat("================================================================\n")
cat("Fin del reporte estadistico\n")
cat("================================================================\n")
sink()

cat(paste("  Reporte guardado:", reporte_path, "\n"))
cat("\n========================================\n")
cat("ANALISIS ESTADISTICO COMPLETADO\n")
cat("========================================\n")
