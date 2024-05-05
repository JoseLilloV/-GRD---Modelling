svrFunction <- function(df_temp){
  library(dplyr)
  library(caret)
  library(tidyverse)
  library(ranger)
  library(MLmetrics)
  library(caretEnsemble)
  library(ecr)
# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 2:120) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las n características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD' al DataFrame
  df$GRD <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD %>%
    createDataPartition(p = 0.8, list = FALSE)  # Crear partición de datos, 80% para entrenamiento
  train.data <- df[training.samples, ]          # Datos de entrenamiento
  test.data <- df[-training.samples, ]           # Datos de prueba
  
  # Configuración para la sintonización de parámetros para SVR
  fitControl <- trainControl(
    method = "repeatedcv",     # Método de validación cruzada (5-fold CV)
    number = 10, # Número de divisiones (pliegues) en la validación cruzada
    repeats = 3,
    verboseIter = FALSE,  # No mostrar información detallada durante la sintonización
  )
  
  # Configuración de la cuadrícula de hiperparámetros para SVR
  Grid <- expand.grid(
    C = seq(0, 2, length = 20),  # Valores de C (hiperparámetro de costo)
    sigma = c(0.001, 0.01, 0.1, 1, 10)  # Valores para el parámetro de ancho de banda sigma
  )
  
  # Entrenar el modelo de SVR
  set.seed(565)
  svr <- train(
    x = train.data[,-ncol(train.data)],
    y = train.data$GRD,
    tuneGrid = Grid,
    method = "svmRadial",
    metric = 'Rsquared',
    trControl = fitControl
  )
  
  # Realizar predicciones con el modelo
  predictions <- predict(svr,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD)
  
  # Almacenar los resultados en el DataFrame
  results_df_svr <- rbind(results_df_svr, data.frame(n = n, C = svr$bestTune$C, sigma = svr$bestTune$sigma, R2 = R2, RMSE = RMSE, MAE = MAE))
}
 
}
