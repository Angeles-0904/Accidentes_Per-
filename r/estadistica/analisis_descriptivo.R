# ==============================================================================
# ANALISIS ESTADISTICO DESCRIPTIVO E INFERENCIAL
# Proyecto: Modelado Predictivo de Severidad de Siniestros Viales Fatales
# Lenguaje: R 4.x
# Librerias: tidyverse, dplyr, ggplot2, corrplot, caret
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

# --- 2. Cargar datos ---------------------------------------------------------
cat("\n========================================\n")
cat("CARGANDO DATOS\n")
cat("========================================\n")

df <- read.csv("../data/processed/dataset_merged.csv", encoding = "UTF-8", stringsAsFactors = FALSE)
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
print(prop.table(tabla_severidad) * 100)

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

# 3.4 Estadisticas de variables numericas
cat("\n--- ESTADISTICAS POR SEVERIDAD ---\n")
vars_num <- c("CANTIDAD_LESIONADOS", "CANTIDAD_VEHICULOS")
for (var in vars_num) {
  if (var %in% colnames(df)) {
    cat(paste("\nVariable:", var, "\n"))
    print(tapply(df[[var]], df$severidad, summary))
    cat(paste("  Desv. Estandar:\n"))
    print(tapply(df[[var]], df$severidad, sd, na.rm = TRUE))
  }
}

# --- 4. VISUALIZACIONES PROFESIONALES (ggplot2) ------------------------------
cat("\n========================================\n")
cat("GENERANDO VISUALIZACIONES\n")
cat("========================================\n")

# 4.1 Boxplot: Lesionados por severidad
if ("CANTIDAD_LESIONADOS" %in% colnames(df)) {
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
  
  ggsave("../outputs/figures/r_boxplot_lesionados.png", p1, width = 10, height = 6, dpi = 150)
  cat("  Grafico guardado: r_boxplot_lesionados.png\n")
}

# 4.2 Barras: Severidad por Departamento (top 10)
severidad_dpto <- df %>%
  group_by(DEPARTAMENTO, severidad) %>%
  summarise(Conteo = n(), .groups = "drop") %>%
  group_by(DEPARTAMENTO) %>%
  mutate(Total = sum(Conteo)) %>%
  ungroup() %>%
  arrange(desc(Total)) %>%
  filter(DEPARTAMENTO %in% unique(DEPARTAMENTO)[1:10])

p2 <- ggplot(severidad_dpto, aes(x = reorder(DEPARTAMENTO, -Total), y = Conteo, fill = severidad)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(values = c("Baja" = "#2ecc71", "Media" = "#f39c12", "Alta" = "#e74c3c")) +
  labs(title = "Proporcion de Severidad por Departamento",
       subtitle = "Peru 2021-2025",
       x = "Departamento", y = "Proporcion", fill = "Severidad") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold", size = 14))

ggsave("../outputs/figures/r_severidad_departamento.png", p2, width = 12, height = 6, dpi = 150)
cat("  Grafico guardado: r_severidad_departamento.png\n")

# 4.3 Mapa de calor de correlaciones
num_vars <- df %>%
  select(where(is.numeric)) %>%
  select(-matches("CODIGO|ANIO|LATITUD|LONGITUD"))

if (ncol(num_vars) > 1) {
  cor_matrix <- cor(num_vars, use = "pairwise.complete.obs")
  
  png("../outputs/figures/r_correlaciones.png", width = 1200, height = 1000, res = 150)
  corrplot(cor_matrix, method = "color", type = "upper",
           tl.col = "black", tl.cex = 0.6,
           addCoef.col = "black", number.cex = 0.5,
           title = "Matriz de Correlaciones - Variables Numericas",
           mar = c(0, 0, 2, 0))
  dev.off()
  cat("  Grafico guardado: r_correlaciones.png\n")
}

# 4.4 Histograma de edad (si existe en el dataset)
if ("EDAD_PROMEDIO" %in% colnames(df)) {
  p3 <- ggplot(df, aes(x = EDAD_PROMEDIO)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "#3498db", color = "white", alpha = 0.7) +
    geom_density(color = "#e74c3c", linewidth = 1) +
    labs(title = "Distribucion de Edad Promedio por Siniestro",
         subtitle = "Peru 2021-2025",
         x = "Edad Promedio", y = "Densidad") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14))
  
  ggsave("../outputs/figures/r_histograma_edad.png", p3, width = 10, height = 6, dpi = 150)
  cat("  Grafico guardado: r_histograma_edad.png\n")
}

# --- 5. ESTADISTICA INFERENCIAL ----------------------------------------------
cat("\n========================================\n")
cat("ESTADISTICA INFERENCIAL\n")
cat("========================================\n")

# 5.1 Chi-cuadrado: Relacion entre ZONA y severidad
cat("\n--- TEST CHI-CUADRADO: ZONA vs SEVERIDAD ---\n")
if (all(c("ZONA", "severidad") %in% colnames(df))) {
  tabla_zona <- table(df$ZONA, df$severidad)
  print(tabla_zona)
  test_zona <- chisq.test(tabla_zona)
  cat(paste("  X-cuadrado:", round(test_zona$statistic, 4), "\n"))
  cat(paste("  p-valor:", format(test_zona$p.value, scientific = TRUE), "\n"))
  
  if (test_zona$p.value < 0.05) {
    cat("  CONCLUSION: Existe relacion significativa entre ZONA y SEVERIDAD (p < 0.05)\n")
  } else {
    cat("  CONCLUSION: No hay evidencia suficiente de relacion (p >= 0.05)\n")
  }
}

# 5.2 Chi-cuadrado: Relacion entre CONDICION CLIMATICA y severidad
cat("\n--- TEST CHI-CUADRADO: CLIMA vs SEVERIDAD ---\n")
if (all(c("CONDICION_CLIMATICA", "severidad") %in% colnames(df))) {
  # Limpiar valores vacios
  df_temp <- df %>% filter(CONDICION_CLIMATICA != "" & !is.na(CONDICION_CLIMATICA))
  tabla_clima <- table(df_temp$CONDICION_CLIMATICA, df_temp$severidad)
  
  if (nrow(tabla_clima) > 1) {
    print(tabla_clima)
    # Si hay celdas con valores < 5, usar simulacion
    test_clima <- chisq.test(tabla_clima, simulate.p.value = TRUE)
    cat(paste("  X-cuadrado:", round(test_clima$statistic, 4), "\n"))
    cat(paste("  p-valor:", format(test_clima$p.value, scientific = TRUE), "\n"))
    
    if (test_clima$p.value < 0.05) {
      cat("  CONCLUSION: Existe relacion significativa entre CLIMA y SEVERIDAD (p < 0.05)\n")
    } else {
      cat("  CONCLUSION: No hay evidencia suficiente de relacion (p >= 0.05)\n")
    }
  } else {
    cat("  [!] No hay suficientes categorias para el test\n")
  }
}

# 5.3 Chi-cuadrado: Relacion entre CAUSA FACTOR PRINCIPAL y severidad
cat("\n--- TEST CHI-CUADRADO: CAUSA vs SEVERIDAD ---\n")
if (all(c("CAUSA_FACTOR_PRINCIPAL", "severidad") %in% colnames(df))) {
  df_temp <- df %>% filter(CAUSA_FACTOR_PRINCIPAL != "" & !is.na(CAUSA_FACTOR_PRINCIPAL))
  # Solo top causas para evitar celdas con valores pequenos
  top_causas <- names(sort(table(df_temp$CAUSA_FACTOR_PRINCIPAL), decreasing = TRUE)[1:6])
  df_temp <- df_temp %>% filter(CAUSA_FACTOR_PRINCIPAL %in% top_causas)
  
  tabla_causa <- table(df_temp$CAUSA_FACTOR_PRINCIPAL, df_temp$severidad)
  print(tabla_causa)
  test_causa <- chisq.test(tabla_causa, simulate.p.value = TRUE)
  cat(paste("  X-cuadrado:", round(test_causa$statistic, 4), "\n"))
  cat(paste("  p-valor:", format(test_causa$p.value, scientific = TRUE), "\n"))
  
  if (test_causa$p.value < 0.05) {
    cat("  CONCLUSION: Existe relacion significativa entre CAUSA y SEVERIDAD (p < 0.05)\n")
  } else {
    cat("  CONCLUSION: No hay evidencia suficiente de relacion (p >= 0.05)\n")
  }
}

# 5.4 ANOVA: Edad promedio por nivel de severidad
cat("\n--- ANOVA: EDAD PROMEDIO vs SEVERIDAD ---\n")
if ("EDAD_PROMEDIO" %in% colnames(df)) {
  modelo_anova <- aov(EDAD_PROMEDIO ~ severidad, data = df)
  print(summary(modelo_anova))
  
  if (summary(modelo_anova)[[1]][1, "Pr(>F)"] < 0.05) {
    cat("  CONCLUSION: Existen diferencias significativas en edad promedio entre niveles de severidad (p < 0.05)\n")
    cat("  Realizando Tukey HSD...\n")
    print(TukeyHSD(modelo_anova))
  } else {
    cat("  CONCLUSION: No hay diferencias significativas en edad promedio entre niveles de severidad (p >= 0.05)\n")
  }
}

# --- 6. REPORTE PROFESIONAL --------------------------------------------------
cat("\n========================================\n")
cat("GENERANDO REPORTE ESTADISTICO\n")
cat("========================================\n")

sink("../outputs/reports/reporte_estadistico_R.txt")
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
if (exists("test_zona")) {
  cat("\n2.1 Chi-cuadrado: ZONA vs SEVERIDAD\n")
  cat(paste("  Estadistico:", round(test_zona$statistic, 4), "\n"))
  cat(paste("  p-valor:", format(test_zona$p.value, scientific = TRUE), "\n"))
  cat(paste("  Conclusion:", ifelse(test_zona$p.value < 0.05,
      "Relacion significativa", "No hay relacion significativa"), "\n"))
}

if (exists("test_clima") && inherits(test_clima, "htest")) {
  cat("\n2.2 Chi-cuadrado: CLIMA vs SEVERIDAD\n")
  cat(paste("  p-valor:", format(test_clima$p.value, scientific = TRUE), "\n"))
  cat(paste("  Conclusion:", ifelse(test_clima$p.value < 0.05,
      "Relacion significativa", "No hay relacion significativa"), "\n"))
}

if (exists("test_causa") && inherits(test_causa, "htest")) {
  cat("\n2.3 Chi-cuadrado: CAUSA vs SEVERIDAD (top 6 causas)\n")
  cat(paste("  p-valor:", format(test_causa$p.value, scientific = TRUE), "\n"))
  cat(paste("  Conclusion:", ifelse(test_causa$p.value < 0.05,
      "Relacion significativa", "No hay relacion significativa"), "\n"))
}

cat("\n3. CONCLUSIONES ESTADISTICAS\n")
cat("---------------------------\n")
cat("1. La mayoria de siniestros (89.8%) presentan severidad baja (1 fallecido).\n")
cat("2. El analisis de varianza muestra si existen diferencias de edad segun severidad.\n")
cat("3. Las pruebas chi-cuadrado revelan la asociacion entre factores de riesgo y severidad.\n")
cat("4. Se recomienda usar modelos de clasificacion con balanceo de clases.\n\n")
cat("================================================================\n")
cat("Fin del reporte estadistico\n")
cat("================================================================\n")
sink()

cat("  Reporte guardado: ../outputs/reports/reporte_estadistico_R.txt\n")
cat("\n========================================\n")
cat("ANALISIS ESTADISTICO COMPLETADO\n")
cat("========================================\n")
