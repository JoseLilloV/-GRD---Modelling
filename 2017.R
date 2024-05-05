# Carga de bibliotecas necesarias
library(dplyr)
library(caret)
library(tidyverse)
library(ranger)
library(MLmetrics)
library(caretEnsemble)

# Establecer una semilla para la reproducibilidad
set.seed(565)

# Abrir archivo CSV
df <- read.csv("datos finales 2017.csv", header = TRUE, sep = ";")[-1]  # Leer datos y omitir la primera columna


################################################################################
#                           PREPARACION DE DATOS                               #
################################################################################

# Filtrar y mantener solo los registros con valores en la columna GRD2017
df$GRD2017 <- gsub(",", ".", df$GRD2017)  # Reemplazar comas por puntos
df$GRD2017 <- na_if(df$GRD2017, "")       # Reemplazar cadenas vacías por NA
df$GRD2017 <- as.numeric(df$GRD2017)      # Convertir a tipo numérico
df <- df[complete.cases(df$GRD2017), ]    # Eliminar registros con valores NA en GRD2017

#Guardar variable dependiente
GRD <- df$GRD2017

## Remplazar los NA por 0 en todo el DataFrame
df <- replace(df, is.na(df), 0)


################################################################################
#                        PREPROCESAMIENTO DE DATOS                             #
################################################################################

# Identificación y eliminación de características con varianza cercana a cero
nzv <- nearZeroVar(df, saveMetrics = TRUE, freqCut = 100/2)
nzv["Variables"] <- row.names(nzv)
descritiva_nzv <- nzv %>%
  filter(nzv == TRUE) %>%
  select(Variables, freqRatio, percentUnique)
retirados_nzv <- descritiva_nzv$Variables
df <- select(df, -retirados_nzv)

# Identificar y eliminar predictores correlacionados
df_pre_numeric <- df %>%
  select_if(is.numeric)

corr_matrix <- cor(df_pre_numeric, use = "pairwise.complete.obs")
highlyCorDescr <- findCorrelation(corr_matrix, cutoff = 0.75)
retirados_cor <- colnames(df_pre_numeric[, highlyCorDescr])
df <- df_pre_numeric[, -highlyCorDescr]

# Normalización de los datos
preproc <- preProcess(df, method = c("range"))
df <- predict(preproc, df)
df_temp <- select(df, -GRD2017)

# Selección de características mediante eliminación recursiva utilizando Random Forest
subsets <- 1:length(df)
# Configurar el control de RFE
rfe_control = rfeControl(functions = treebagFuncs,  # Funciones de Treebag para la selección de características
                        method = "cv",            # Método de validación cruzada (en este caso, Cross-Validation)
                        number = 5,               # Número de particiones en la validación cruzada
                        returnResamp = "all",     # Devolver todos los resultados de remuestreo
                        verbose = TRUE            # Mostrar información detallada durante el proceso
                      )

set.seed(565)
rfe_result = rfe(x= df_temp,
                 y= GRD,
                 sizes=subsets,
                 metric="RMSE",
                 rfeControl = rfe_control)

# Imprimir las características seleccionadas
predictors(rfe_result)

 save(rfe_result,file="rfe_result_rf_2016_grd45.RData")
# load('rfe_result_rf_2015_grd208.RData')


################################################################################
#                               RANDOM FOREST                                  #
################################################################################

# Crear un DataFrame para almacenar los resultados de Random Forest
results_df_rf <- data.frame(n = numeric(0), mtry = numeric(0), minnodesize = numeric(0), splitrule = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 2:113) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las n características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD2017' al DataFrame
  df$GRD2017 <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD2017 %>%
    createDataPartition(p = 0.8, list = FALSE)  # Crear partición de datos, 80% para entrenamiento
  train.data <- df[training.samples, ]          # Datos de entrenamiento
  test.data <- df[-training.samples, ]           # Datos de prueba
  
  # Configuración para la sintonización de parámetros
  fitControl <- trainControl(
    method = "cv",     # Método de validación cruzada (5-fold CV)
    number = 5,        # Número de divisiones (pliegues) en la validación cruzada
    verboseIter = FALSE,  # No mostrar información detallada durante la sintonización
    returnData = FALSE
  )
  
  # Configuración de la cuadrícula de hiperparámetros para Random Forest (RF)
  Grid <- expand.grid(
    mtry = c(2:10),  # Rango de valores para el número de características a considerar
    min.node.size = c(2:10),  # Rango de valores para el tamaño mínimo del nodo terminal
    splitrule = c("variance", "extratrees", "maxstat")  # Diferentes reglas de división
  )
  
  # Entrenar el modelo de Random Forest
  set.seed(565)
  rf <- train(
    x = train.data[,-ncol(train.data)],
    y = train.data$GRD2017,
    tuneGrid = Grid,
    tuneLength = 150,  # Número de combinaciones a probar
    method = "ranger",
    metric = 'Rsquared',
    importance = 'impurity',
    trControl = fitControl
  )
  
  # Realizar predicciones con el modelo
  predictions <- predict(rf$finalModel, data = test.data)$predictions
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD2017)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD2017)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD2017)
  
  # Almacenar los resultados en el DataFrame
  results_df_rf <- rbind(results_df_rf, data.frame(n = n, mtry = rf$bestTune$mtry, minnodesize = rf$bestTune$min.node.size, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados
write.csv(results_df_rf, file = "result_rf.txt", row.names = TRUE)

################################################################################
#                     Regresión vectorial por soporte                          #
################################################################################

# Crear un DataFrame para almacenar los resultados de SVR
results_df_svr <- data.frame(n = numeric(0), C = numeric(0), sigma = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 2:113) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las n características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD2017' al DataFrame
  df$GRD2017 <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD2017 %>%
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
    y = train.data$GRD2017,
    tuneGrid = Grid,
    method = "svmRadial",
    metric = 'Rsquared',
    trControl = fitControl
  )
  
  # Realizar predicciones con el modelo
  predictions <- predict(svr,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD2017)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD2017)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD2017)
  
  # Almacenar los resultados en el DataFrame
  results_df_svr <- rbind(results_df_svr, data.frame(n = n, C = svr$bestTune$C, sigma = svr$bestTune$sigma, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados de svr en un archivo CSV
write.csv(results_df_svr, file = "result_svr.txt", row.names = TRUE)



################################################################################
#                         k vecinos mas cercanos                               #
################################################################################

# Crear un DataFrame para almacenar los resultados de K Vecinos Más Cercanos (KNN)
results_df_knn <- data.frame(n = numeric(0), k = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 2:113) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las n características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD2017' al DataFrame
  df$GRD2017 <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD2017 %>%
    createDataPartition(p = 0.8, list = FALSE)  # Crear partición de datos, 80% para entrenamiento
  train.data <- df[training.samples, ]          # Datos de entrenamiento
  test.data <- df[-training.samples, ]           # Datos de prueba
  
  # Configuración para la sintonización de parámetros para KNN
  fitControl <- trainControl(
    method = "repeatedcv",     # Método de validación cruzada (5-fold CV)
    number = 10,              # Número de divisiones (pliegues) en la validación cruzada
    repeats = 3,             # Número de repeticiones
  )
  
  # Entrenar el modelo de KNN
  set.seed(565)
  knn <- train(
    x = train.data[,-ncol(train.data)],
    y = train.data$GRD2017,
    method = "knn",
    tuneLength = 10,         # Número de combinaciones a probar
    metric = 'RMSE',
    trControl = fitControl,
  )
  
  # Realizar predicciones con el modelo KNN
  predictions <- predict(knn,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD2017)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD2017)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD2017)
  
  # Almacenar los resultados en el DataFrame
  results_df_knn <- rbind(results_df_knn, data.frame(n = n, k = knn$bestTune$k, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados de KNN en un archivo CSV
write.csv(results_df_knn, file = "result_knn.txt", row.names = TRUE)

################################################################################
#                          K Vecinos Ponderados                                #
################################################################################

# Crear un DataFrame para almacenar los resultados de KKNN (K Vecinos Ponderados)
results_df_kknn <- data.frame(n = numeric(0), k = numeric(0), distance = numeric(0), kernel = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 2:113) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las n características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD2017' al DataFrame
  df$GRD2017 <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD2017 %>%
    createDataPartition(p = 0.8, list = FALSE)  # Crear partición de datos, 80% para entrenamiento
  train.data <- df[training.samples, ]          # Datos de entrenamiento
  test.data <- df[-training.samples, ]           # Datos de prueba
  
  # Configuración para la sintonización de parámetros para KKNN
  fitControl <- trainControl(
    method = "repeatedcv",     # Método de validación cruzada (5-fold CV)
    number = 10,              # Número de divisiones (pliegues) en la validación cruzada
    repeats = 3,              # Número de repeticiones
  )
  
  # Entrenar el modelo de KKNN
  set.seed(565)
  kknn <- train(
    x = train.data[,-ncol(train.data)],
    y = train.data$GRD2017,
    method = "kknn",
    tuneLength = 10,         # Número de combinaciones a probar
    metric = 'RMSE',
    trControl = fitControl,
  )
  
  # Realizar predicciones con el modelo KKNN
  predictions <- predict(kknn$finalModel,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD2017)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD2017)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD2017)
  
  # Almacenar los resultados en el DataFrame
  results_df_kknn <- rbind(results_df_kknn, data.frame(n = n, k = kknn$bestTune$kmax, distance = kknn$bestTune$distance, kernel = kknn$bestTune$kernel, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados de KKNN en un archivo CSV
write.csv(results_df_kknn, file = "result_kknn.txt", row.names = TRUE)



################################################################################
#                       Gradient Boosting Machine                              #
################################################################################

# Crear un DataFrame para almacenar los resultados de GBM (Gradient Boosting Machine)
results_df_gbm <- data.frame(n = numeric(0), n.trees = numeric(0), interaction.depth = numeric(0), shrinkage = numeric(0), n.minobsinnode = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 2:113) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las 20 características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD2017' al DataFrame
  df$GRD2017 <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD2017 %>%
    createDataPartition(p = 0.8, list = FALSE)  # Crear partición de datos, 80% para entrenamiento
  train.data <- df[training.samples, ]          # Datos de entrenamiento
  test.data <- df[-training.samples, ]           # Datos de prueba
  
  # Configuración para la sintonización de parámetros para GBM
  fitControl <- trainControl(
    method = "repeatedcv",     # Método de validación cruzada (5-fold CV)
    number = 10,              # Número de divisiones (pliegues) en la validación cruzada
    repeats = 3,              # Número de repeticiones
  )
  
  # Configuración de la cuadrícula de hiperparámetros para GBM
  Grid <- expand.grid(
    interaction.depth = 1:2,  # Profundidad de interacción
    shrinkage = 0.1,          # Factor de reducción (shrinkage)
    n.trees = c(10, 50, 100), # Número de árboles
    n.minobsinnode = 10       # Mínimo de observaciones en un nodo terminal
  )
  
  # Entrenar el modelo de GBM
  set.seed(565)
  gbm <- train(
    x = train.data[,-ncol(train.data)],
    y = train.data$GRD2017,
    method = "gbm",
    tuneGrid = Grid,
    tuneLength = 10,         # Número de combinaciones a probar
    metric = 'RMSE',
    trControl = fitControl
  )
  
  # Realizar predicciones con el modelo GBM
  predictions <- predict(gbm,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD2017)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD2017)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD2017)
  
  # Almacenar los resultados en el DataFrame
  results_df_gbm <- rbind(results_df_gbm, data.frame(n = n, n.trees = gbm$bestTune$n.trees, interaction.depth = gbm$bestTune$interaction.depth, 
                                                     shrinkage = gbm$bestTune$shrinkage, n.minobsinnode = gbm$bestTune$n.minobsinnode , R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados de GBM en un archivo CSV
write.csv(results_df_gbm, file = "result_gbm.txt", row.names = TRUE)


################################################################################
#                         Emsemble  model lm + rf                              #
################################################################################
# Crear un DataFrame para almacenar los resultados del modelo de Stacking con LM y RF
results_df_stacking_lmrf <- data.frame(n = numeric(0), lm_intercept = character(0), mtry = numeric(0), splitrule = character(0), min.node.size = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 10:113) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las 20 características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD2017' al DataFrame
  df$GRD2017 <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD2017 %>%
    createDataPartition(p = 0.8, list = FALSE)  # Crear partición de datos, 80% para entrenamiento
  train.data <- df[training.samples, ]          # Datos de entrenamiento
  test.data <- df[-training.samples, ]           # Datos de prueba
  
  # Configuración para la sintonización de parámetros
  fitControl <- trainControl(
    method = "cv",  # 5-fold CV
    number = 5,     # Número de divisiones (pliegues) en la validación cruzada
    verboseIter = TRUE,
    returnData = FALSE,
    trim = TRUE,
    savePredictions = "final"
  )
  
  # Entrenar modelos LM y RF
  set.seed(565)
  model_list_complete <- caretList(
    x = train.data[,-ncol(train.data)],
    y = train.data$GRD2017,
    trControl = fitControl,
    metric = "RMSE",
    tuneList = list(
      lm = caretModelSpec(method = "lm"),
      rf = caretModelSpec(method = "ranger", 
                          tuneGrid = data.frame(.mtry=c(2:10),
                                                .splitrule =  "variance",
                                                .min.node.size=c(2:10)))
    )
  )
  
  # Configuración de la cuadrícula de hiperparámetros para modelo lineal (LM)
  grid  <- data.frame(intercept = c(TRUE, FALSE))
  
  # Entrenar el modelo de Stacking con LM y RF
  set.seed(565)
  model <- caretStack(
    model_list_complete, 
    trControl = fitControl,
    metric = "RMSE",
    method = "lm",
    tuneGrid = grid
  )
  
  # Realizar predicciones con el modelo de Stacking
  predictions <- predict(model, newdata = test.data)
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD2017)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD2017)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD2017)
  
  # Almacenar los resultados en el DataFrame
  results_df_stacking_lmrf <- rbind(results_df_stacking_lmrf, data.frame(n = n, lm_intercept = model$models$lm$bestTune$intercept, mtry = model$models$rf$bestTune$mtry, splitrule = model$models$rf$bestTune$splitrule, min.node.size = model$models$rf$bestTune$min.node.size, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados del modelo de Stacking en un archivo CSV
write.csv(results_df_stacking_lmrf, file = "result_stacking_lm_rf.txt", row.names = TRUE)
################################################################################
#                         Emsemble  model lm + knn                             #
################################################################################
# Crear un DataFrame para almacenar los resultados del modelo de Stacking con LM y RF
results_df_stacking_lmknn <- data.frame(n = numeric(0), lm_intercept = character(0), k = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 10:113) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las 20 características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD2017' al DataFrame
  df$GRD2017 <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD2017 %>%
    createDataPartition(p = 0.8, list = FALSE)  # Crear partición de datos, 80% para entrenamiento
  train.data <- df[training.samples, ]          # Datos de entrenamiento
  test.data <- df[-training.samples, ]           # Datos de prueba
  
  # Configuración para la sintonización de parámetros
  fitControl <- trainControl(
    method = "repeatedcv",  # 5-fold CV
    number = 10,     # Número de divisiones (pliegues) en la validación cruzada
    repeats = 3,
    verboseIter = TRUE,
    returnData = FALSE,
    trim = TRUE,
    savePredictions = "final"
  )
  
  # Entrenar modelos LM y RF
  set.seed(565)
  model_list_complete <- caretList(
    x = train.data[,-ncol(train.data)],
    y = train.data$GRD2017,
    trControl = fitControl,
    metric = "RMSE",
    tuneList = list(
      lm = caretModelSpec(method = "lm"),
      knn = caretModelSpec(method = "knn",  
                           tuneLength = 10)
    )
  )
  
  # Configuración de la cuadrícula de hiperparámetros para modelo lineal (LM)
  grid  <- data.frame(intercept = c(TRUE, FALSE))
  
  # Entrenar el modelo de Stacking con LM y RF
  set.seed(565)
  model <- caretStack(
    model_list_complete, 
    trControl = fitControl,
    metric = "RMSE",
    method = "knn",
    tuneLength= 10,
    # tuneGrid = grid,  
  )
  
  # Realizar predicciones con el modelo de Stacking
  predictions <- predict(model, newdata = test.data)
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD2017)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD2017)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD2017)
  
  # Almacenar los resultados en el DataFrame
  results_df_stacking_lmknn <- rbind(results_df_stacking_lmknn, data.frame(n = n, lm_intercept = model$models$lm$bestTune$intercept, k = model$models$knn$bestTune$k, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados del modelo de Stacking en un archivo CSV
write.csv(results_df_stacking_lmknn, file = "result_stacking_lm_knn.txt", row.names = TRUE)



################################################################################
#                               Corelacion                                     #
################################################################################


correlacion_lista <- matrix(ncol = 0, nrow = 27)

for(i in c(1:27)){
  resultado <- cor(df[row.names(varImp(rfe_result))[i]], df$GRD2017)
  correlacion_lista[i] = resultado
  
}
