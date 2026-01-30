############################################
# PROYECTO ML – LALIGA (R)
# Carga desde Excel + Preprocesado completo
############################################

# ===============================
# 1. LIBRERÍAS
# ===============================

library(readxl)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(caret)
library(corrplot)

# ===============================
# 2. CARGA DEL DATASET (EXCEL)
# ===============================

df <- read_excel(
  "C:/Users/jaime/OneDrive/Escritorio/Personal/Universidad Europea/R/Tareas/Actividad Machine Learning/resultados_la_liga.xlsx"
)

# Normalizar nombres de columnas
names(df) <- names(df) %>%
  trimws() %>%
  tolower()
df

# ===============================
# 3. FECHA → YEAR / MONTH / HOUR
# ===============================

df <- df %>%
  mutate(
    matchdate = as.POSIXct(
      matchdate,
      format = "%d-%m-%y %H:%M",
      tz = "Europe/Madrid"
    ),
    year  = year(matchdate),
    month = month(matchdate),
    hour  = hour(matchdate)
  )

# Crear columnas útiles
df <- df %>%
  mutate(
    home_win = ifelse(ftr == "H", 1, 0),
    away_win = ifelse(ftr == "A", 1, 0),
    draw = ifelse(ftr == "D", 1, 0)
  )

# --- Análisis exploratorio ---

# 1. Media de goles por equipo (local y visitante)
goles_media <- df %>%
  group_by(hometeam) %>%
  summarise(
    goles_marcados = mean(fthg),
    goles_recibidos = mean(ftag)
  ) %>%
  arrange(desc(goles_marcados))

# 2. Porcentaje de victorias
victorias <- df %>%
  group_by(hometeam) %>%
  summarise(
    total_partidos = n(),
    victorias_local = sum(home_win),
    victorias_fuera = sum(away_win)
  ) %>%
  mutate(
    pct_victorias_local = victorias_local / total_partidos * 100,
    pct_victorias_fuera = victorias_fuera / total_partidos * 100
  )

# 3. Goles recibidos fuera de casa
goles_fuera <- df %>%
  group_by(awayteam) %>%
  summarise(goles_recibidos_fuera = mean(fthg)) %>%
  arrange(desc(goles_recibidos_fuera))

# 4. Tendencias de victorias por mes fuera de casa
victorias_mes <- df %>%
  group_by(month) %>%
  summarise(victorias_fuera = sum(away_win),
            total_partidos_fuera = n(),
            pct_victorias_fuera = victorias_fuera / total_partidos_fuera * 100)

# --- Gráficos con ggplot2 ---

# Media de goles marcados por equipo
ggplot(goles_media, aes(x = reorder(hometeam, goles_marcados), y = goles_marcados)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Media de goles marcados por equipo en casa",
       x = "Equipo", y = "Goles marcados")

# Porcentaje de victorias fuera de casa
ggplot(victorias_mes, aes(x = month, y = pct_victorias_fuera)) +
  geom_line(group = 1, color = "red", size = 1.2) +
  geom_point(color = "red", size = 2) +
  labs(title = "Porcentaje de victorias fuera de casa por mes",
       x = "Mes", y = "% Victorias fuera de casa")

# Goles recibidos fuera de casa por equipo
ggplot(goles_fuera, aes(x = reorder(awayteam, goles_recibidos_fuera), y = goles_recibidos_fuera)) +
  geom_col(fill = "orange") +
  coord_flip() +
  labs(title = "Media de goles recibidos fuera de casa por equipo",
       x = "Equipo", y = "Goles recibidos")

# --- Identificación de variables relevantes ---
# Aquí se puede usar correlación simple entre goles, victorias y otras métricas
correlaciones <- df %>%
  select(fthg, ftag, home_win, away_win) %>%
  cor()

print(correlaciones)

# Entrenamiento del modelo y Evaluación del modelo: 
# Librerías necesarias
library(caret)
library(randomForest)
library(pROC)

# --- Preparación de los datos para el modelo ---

# Convertimos la variable target: 1 si gana local, 0 si no
df <- df %>%
  mutate(
    local_win = ifelse(ftr == "H", 1, 0)
  )

# Selección de variables predictoras (puedes añadir más)
# Usaremos goles de local/visitante y goles de medio tiempo como features
features <- df %>%
  select(fthg, ftag, `1hhg`, `1hag`, `2hhg`, `2hag`, month, hour)

# Convertimos variables categóricas a factor
features$month <- as.factor(features$month)
features$hour <- as.factor(features$hour)

# Definimos la variable target
target <- as.factor(df$home_win)

# Dividimos en conjunto de entrenamiento y prueba (70%-30%)
set.seed(123)
train_index <- createDataPartition(target, p = 0.7, list = FALSE)
X_train <- features[train_index, ]
X_test  <- features[-train_index, ]
y_train <- target[train_index]
y_test  <- target[-train_index]

# --- Entrenamiento del modelo Random Forest ---
set.seed(123)
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

# Mostrar importancia de variables
print(rf_model)
varImpPlot(rf_model)

# --- Predicciones y evaluación ---
pred_train <- predict(rf_model, X_train)
pred_test  <- predict(rf_model, X_test)

# Matriz de confusión
confusionMatrix(pred_test, y_test)

# Curva ROC y AUC
prob_test <- predict(rf_model, X_test, type = "prob")[,2]
roc_obj <- roc(as.numeric(y_test), prob_test)
plot(roc_obj, main = "Curva ROC - Random Forest")
auc(roc_obj)

# --- Ajuste de hiperparámetros con caret ---
set.seed(123)
tune_grid <- expand.grid(
  .mtry = c(2,3,4),
  .splitrule = "gini",
  .min.node.size = c(1,3,5)
)

train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

rf_caret <- train(
  x = X_train,
  y = y_train,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "ROC"
)

print(rf_caret)

# ===============================
# APLICACIÓN PRÁCTICA DEL MODELO
# ===============================
# En esta sección aplicamos el modelo entrenado para:
# 1) Predecir el resultado de partidos
# 2) Clasificar equipos según su rendimiento histórico
# 3) Guardar los resultados en un archivo CSV
# 4) Visualizar las predicciones para facilitar su interpretación

library(dplyr)
library(ggplot2)

# -------------------------------------------------------
# 1. PREDICCIÓN DEL RESULTADO DE PARTIDOS
# -------------------------------------------------------
# Utilizamos el conjunto de test (datos no vistos por el modelo)
# para simular un caso real de predicción de resultados

# Predicción de la clase (0 = no gana local, 1 = gana local)
predicciones <- predict(rf_model, X_test)

# Predicción de probabilidades (útil para interpretación)
probabilidades <- predict(rf_model, X_test, type = "prob")[,2]

# Creamos un dataframe con los resultados reales y predichos
resultados_pred <- df %>%
  slice(-train_index) %>%
  select(matchdate, hometeam, awayteam, ftr) %>%
  mutate(
    resultado_real = ifelse(ftr == "H", "Victoria Local", "No Victoria Local"),
    prediccion_modelo = ifelse(predicciones == 1, "Victoria Local", "No Victoria Local"),
    prob_victoria_local = probabilidades
  )

head(resultados_pred)
# -------------------------------------------------------
# 2. INTERPRETACIÓN PRÁCTICA
# -------------------------------------------------------
# Este dataframe nos permite responder preguntas como:
# - ¿Qué partidos predijo correctamente el modelo?
# - ¿Con qué probabilidad se estima que el equipo local gane?

# Ejemplo: partidos donde el modelo tiene alta confianza (>70%)
predicciones_confianza_alta <- resultados_pred %>%
  filter(prob_victoria_local > 0.7)

head(predicciones_confianza_alta)
# -------------------------------------------------------
# 3. CLASIFICACIÓN DE EQUIPOS SEGÚN RENDIMIENTO HISTÓRICO
# -------------------------------------------------------
# Clasificamos los equipos usando la probabilidad media
# de victoria local predicha por el modelo

ranking_equipos <- resultados_pred %>%
  group_by(hometeam) %>%
  summarise(
    prob_media_victoria = mean(prob_victoria_local),
    partidos_analizados = n()
  ) %>%
  arrange(desc(prob_media_victoria))

head(ranking_equipos)


# -------------------------------------------------------
# 4. GUARDAR RESULTADOS EN ARCHIVOS CSV
# -------------------------------------------------------
# Guardamos las predicciones para su uso posterior
# (documentación, informes o análisis externo)

write.csv(resultados_pred,
          "predicciones_partidos.csv",
          row.names = FALSE)

write.csv(ranking_equipos,
          "ranking_equipos_rendimiento.csv",
          row.names = FALSE)

# -------------------------------------------------------
# 5. VISUALIZACIÓN DE LAS PREDICCIONES
# -------------------------------------------------------
# Gráfico de probabilidades de victoria local por partido

ggplot(resultados_pred,
       aes(x = reorder(hometeam, prob_victoria_local),
           y = prob_victoria_local)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Probabilidad estimada de victoria local por equipo",
    x = "Equipo local",
    y = "Probabilidad de victoria"
  )

# -------------------------------------------------------
# 6. VISUALIZACIÓN DEL RANKING DE EQUIPOS
# -------------------------------------------------------
# Ranking de equipos según su rendimiento histórico predicho

ggplot(ranking_equipos,
       aes(x = reorder(hometeam, prob_media_victoria),
           y = prob_media_victoria)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(
    title = "Clasificación de equipos según rendimiento histórico",
    x = "Equipo",
    y = "Probabilidad media de victoria local"
  )
