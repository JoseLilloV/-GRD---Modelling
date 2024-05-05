# Carga de bibliotecas necesarias
library(dplyr)
library(caret)
library(tidyverse)
library(ranger)
library(MLmetrics)
library(caretEnsemble)
library(ecr)

# Establecer una semilla para la reproducibilidad
set.seed(565)

# Abrir archivo CSV
df <- read.csv("datosfinalescompletos.csv", header = TRUE, sep = ";")[-1]  # Leer datos y omitir la primera columna


################################################################################
#                           PREPARACION DE DATOS                               #
################################################################################

# Filtrar y mantener solo los registros con valores en la columna GRD
df$GRD <- gsub(",", ".", df$GRD)  # Reemplazar comas por puntos
df$GRD <- na_if(df$GRD, "")       # Reemplazar cadenas vacías por NA
df$GRD <- as.numeric(df$GRD)      # Convertir a tipo numérico
df <- df[complete.cases(df$GRD), ]    # Eliminar registros con valores NA en GRD

#Guardar variable dependiente
GRD <- df$GRD

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
df_temp <- select(df, -GRD)

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

# save(rfe_result,file="rfe_result_rf_general_grd162.RData")
# load('rfe_result_rf_2014_grd10.RData.RData')


################################################################################
#                               RANDOM FOREST                                  #
################################################################################

# Crear un DataFrame para almacenar los resultados de Random Forest
results_df_rf <- data.frame(n = numeric(0), mtry = numeric(0), minnodesize = numeric(0), splitrule = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

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
    x = select(train.data, -GRD),
    y = train.data$GRD,
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
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD)
  
  # Almacenar los resultados en el DataFrame
  results_df_rf <- rbind(results_df_rf, data.frame(n = n, mtry = rf$bestTune$mtry, minnodesize = rf$bestTune$min.node.size, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados
write.csv(results_df_rf, file = "result_rf.txt", row.names = TRUE)

# Calcular Hiper volumen
rf <- select(results_df_rf, n, R2, RMSE, MAE)
#rf$R2[rf$R2<0.001] <- 0.001

rf$R2 <- 1/rf$R2

preproc <- preProcess(select(rf, -n), method = c("range"))
rf <- predict(preproc, rf)
rf_matrix <- as.matrix(select(rf, -n))
resultados_rf = matrix(ncol = 0, nrow = 119)


for (i in c(1:119)) {
  resultado=computeHV(as.matrix(rf_matrix[i,]), ref.point = c(2,2,2))
  resultados_rf[i]= resultado
}

print("MEJOR MODELO RF CONSIDERANDO HIPERVOLUMEN")
print( rf[which.max(resultados_rf),])



################################################################################
#                     Regresión vectorial por soporte                          #
################################################################################

# Crear un DataFrame para almacenar los resultados de SVR
results_df_svr <- data.frame(n = numeric(0), C = numeric(0), sigma = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

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

# Guardar Resultados de svr en un archivo CSV
write.csv(results_df_svr, file = "result_svr.txt", row.names = TRUE)

svr <- select(results_df_svr, n, R2, RMSE, MAE)
svr$R2[svr$R2<0.001] <- 0.001

svr$R2 <- 1/svr$R2

preproc <- preProcess(select(svr, -n), method = c("range"))
svr <- predict(preproc, svr)
svr_matrix <- as.matrix(select(svr, -n))
resultados_svr= matrix(ncol = 0, nrow = 119)


for (i in c(1:119)) {
  resultado=computeHV(as.matrix(svr_matrix[i,]), ref.point = c(2,2,2))
  resultados_svr[i]= resultado
}

print("MEJOR MODELO SVR CONSIDERANDO HIPERVOLUMEN")
print( svr[which.max(resultados_svr),])


################################################################################
#                         k vecinos mas cercanos                               #
################################################################################

# Crear un DataFrame para almacenar los resultados de K Vecinos Más Cercanos (KNN)
results_df_knn <- data.frame(n = numeric(0), k = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

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
  
  # Configuración para la sintonización de parámetros para KNN
  fitControl <- trainControl(
    method = "repeatedcv",     # Método de validación cruzada (5-fold CV)
    number = 10,              # Número de divisiones (pliegues) en la validación cruzada
    repeats = 3,             # Número de repeticiones
  )
  
  # Entrenar el modelo de KNN
  set.seed(565)
  knn <- train(
    x = select(train.data, -GRD),
    y = train.data$GRD,
    method = "knn",
    tuneLength = 10,         # Número de combinaciones a probar
    metric = 'RMSE',
    trControl = fitControl,
  )
  
  # Realizar predicciones con el modelo KNN
  predictions <- predict(knn,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD)
  
  # Almacenar los resultados en el DataFrame
  results_df_knn <- rbind(results_df_knn, data.frame(n = n, k = knn$bestTune$k, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados de KNN en un archivo CSV
write.csv(results_df_knn, file = "result_knn.txt", row.names = TRUE)


knn <- select(results_df_knn, n, R2, RMSE, MAE)
knn$R2[knn$R2<0.001] <- 0.001

knn$R2 <- 1/knn$R2

preproc <- preProcess(select(knn, -n), method = c("range"))
knn <- predict(preproc, knn)
knn_matrix <- as.matrix(select(knn, -n))
resultados_knn= matrix(ncol = 0, nrow = 119)


for (i in c(1:119)) {
  resultado=computeHV(as.matrix(knn_matrix[i,]), ref.point = c(2,2,2))
  resultados_knn[i]= resultado
}

print("MEJOR MODELO KNN CONSIDERANDO HIPERVOLUMEN")
print( knn[which.max(resultados_knn),])

################################################################################
#                          K Vecinos Ponderados                                #
################################################################################

# Crear un DataFrame para almacenar los resultados de KKNN (K Vecinos Ponderados)
results_df_kknn <- data.frame(n = numeric(0), k = numeric(0), distance = numeric(0), kernel = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

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
  
  # Configuración para la sintonización de parámetros para KKNN
  fitControl <- trainControl(
    method = "repeatedcv",     # Método de validación cruzada (5-fold CV)
    number = 10,              # Número de divisiones (pliegues) en la validación cruzada
    repeats = 3,              # Número de repeticiones
  )
  
  # Entrenar el modelo de KKNN
  set.seed(565)
  kknn <- train(
    x = select(train.data, -GRD),
    y = train.data$GRD,
    method = "kknn",
    tuneLength = 10,         # Número de combinaciones a probar
    metric = 'RMSE',
    trControl = fitControl,
  )
  
  # Realizar predicciones con el modelo KKNN
  predictions <- predict(kknn$finalModel,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD)
  
  # Almacenar los resultados en el DataFrame
  results_df_kknn <- rbind(results_df_kknn, data.frame(n = n, k = kknn$bestTune$kmax, distance = kknn$bestTune$distance, kernel = kknn$bestTune$kernel, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados de KKNN en un archivo CSV
write.csv(results_df_kknn, file = "result_kknn.txt", row.names = TRUE)


kknn <- select(results_df_kknn, n, R2, RMSE, MAE)
kknn$R2[kknn$R2<0.001] <- 0.001

kknn$R2 <- 1/kknn$R2

preproc <- preProcess(select(kknn, -n), method = c("range"))
kknn <- predict(preproc, kknn)
kknn_matrix <- as.matrix(select(kknn, -n))
resultados_kknn= matrix(ncol = 0, nrow = 119)


for (i in c(1:112)) {
  resultado=computeHV(as.matrix(kknn_matrix[i,]), ref.point = c(2,2,2))
  resultados_kknn[i]= resultado
}

print("MEJOR MODELO kKNN CONSIDERANDO HIPERVOLUMEN")
print( kknn[which.max(resultados_kknn),])


################################################################################
#                       Gradient Boosting Machine                              #
################################################################################

# Crear un DataFrame para almacenar los resultados de GBM (Gradient Boosting Machine)
results_df_gbm <- data.frame(n = numeric(0), n.trees = numeric(0), interaction.depth = numeric(0), shrinkage = numeric(0), n.minobsinnode = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 2:120) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las 20 características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD' al DataFrame
  df$GRD <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD %>%
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
    y = train.data$GRD,
    method = "gbm",
    tuneGrid = Grid,
    tuneLength = 10,         # Número de combinaciones a probar
    metric = 'RMSE',
    trControl = fitControl
  )
  
  # Realizar predicciones con el modelo GBM
  predictions <- predict(gbm,  test.data[,-ncol(train.data)])
  
  # Calcular métricas de evaluación
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD)
  
  # Almacenar los resultados en el DataFrame
  results_df_gbm <- rbind(results_df_gbm, data.frame(n = n, n.trees = gbm$bestTune$n.trees, interaction.depth = gbm$bestTune$interaction.depth, 
                                                     shrinkage = gbm$bestTune$shrinkage, n.minobsinnode = gbm$bestTune$n.minobsinnode , R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados de GBM en un archivo CSV
write.csv(results_df_gbm, file = "result_gbm.txt", row.names = TRUE)

gbm <- select(results_df_gbm, n, R2, RMSE, MAE)
gbm$R2[gbm$R2<0.001] <- 0.001

gbm$R2 <- 1/gbm$R2

preproc <- preProcess(select(gbm, -n), method = c("range"))
gbm <- predict(preproc, gbm)
gbm_matrix <- as.matrix(select(gbm, -n))
resultados_gbm= matrix(ncol = 0, nrow = 119)


for (i in c(1:119)) {
  resultado=computeHV(as.matrix(gbm_matrix[i,]), ref.point = c(2,2,2))
  resultados_gbm[i]= resultado
}

print("MEJOR MODELO GBM CONSIDERANDO HIPERVOLUMEN")
print( gbm[which.max(resultados_gbm),])


################################################################################
#                         Emsemble  model lm + rf                              #
################################################################################
# Crear un DataFrame para almacenar los resultados del modelo de Stacking con LM y RF
results_df_stacking_lmrf <- data.frame(n = numeric(0), lm_intercept = character(0), mtry = numeric(0), splitrule = character(0), min.node.size = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 10:120) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las 20 características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD' al DataFrame
  df$GRD <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD %>%
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
    y = train.data$GRD,
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
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD)
  
  # Almacenar los resultados en el DataFrame
  results_df_stacking_lmrf <- rbind(results_df_stacking_lmrf, data.frame(n = n, lm_intercept = model$models$lm$bestTune$intercept, mtry = model$models$rf$bestTune$mtry, splitrule = model$models$rf$bestTune$splitrule, min.node.size = model$models$rf$bestTune$min.node.size, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados del modelo de Stacking en un archivo CSV
write.csv(results_df_stacking_lmrf, file = "result_stacking_lm_rf.txt", row.names = TRUE)

lmrf <- select(results_df_stacking_lmrf, n, R2, RMSE, MAE)
lmrf$R2[lmrf$R2<0.001] <- 0.001
lmrf$R2 <- 1/lmrf$R2

preproc <- preProcess(select(lmrf, -n), method = c("range"))
lmrf <- predict(preproc, lmrf)
lmrf_matrix <- as.matrix(select(lmrf, -n))
resultados_lmrf= matrix(ncol = 0, nrow = 119)


for (i in c(1:111)) {
  resultado=computeHV(as.matrix(lmrf_matrix[i,]), ref.point = c(2,2,2))
  resultados_lmrf[i]= resultado
}

print("MEJOR MODELO lmrf CONSIDERANDO HIPERVOLUMEN")
print( lmrf[which.max(resultados_lmrf),])


################################################################################
#                         Emsemble  model lm + knn                             #
################################################################################
# Crear un DataFrame para almacenar los resultados del modelo de Stacking con LM y RF
results_df_stacking_lmknn <- data.frame(n = numeric(0), lm_intercept = character(0), k = numeric(0), R2 = numeric(0), RMSE = numeric(0), MAE = numeric(0))

# Bucle para ajustar modelos con diferentes números de características (n)
for (n in 10:120) {
  
  # Seleccionar las características (columnas) en df_temp basadas en la importancia de características resultante del RFE (selecciona las 20 características más importantes)
  df <- select(df_temp, row.names(varImp(rfe_result))[1:n])
  
  # Agregar la variable dependiente 'GRD' al DataFrame
  df$GRD <- GRD
  
  # Dividir los datos en conjuntos de entrenamiento y prueba
  set.seed(565)
  training.samples <- df$GRD %>%
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
    y = train.data$GRD,
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
  RMSE = RMSE(y_pred = predictions, y_true = test.data$GRD)
  MAE = MAE(y_pred = predictions, y_true = test.data$GRD)
  R2 = R2_Score(y_pred = predictions, y_true = test.data$GRD)
  
  # Almacenar los resultados en el DataFrame
  results_df_stacking_lmknn <- rbind(results_df_stacking_lmknn, data.frame(n = n, lm_intercept = model$models$lm$bestTune$intercept, k = model$models$knn$bestTune$k, R2 = R2, RMSE = RMSE, MAE = MAE))
}

# Guardar Resultados del modelo de Stacking en un archivo CSV
write.csv(results_df_stacking_lmknn, file = "result_stacking_lm_knn.txt", row.names = TRUE)


lmknn <- select(results_df_stacking_lmknn, n, R2, RMSE, MAE)

lmknn$R2[lmknn$R2<0.001] <- 0.001
lmknn$R2 <- 1/lmknn$R2

preproc <- preProcess(select(lmknn, -n), method = c("range"))
lmknn <- predict(preproc, lmknn)
lmknn_matrix <- as.matrix(select(lmknn, -n))
resultados_lmknn= matrix(ncol = 0, nrow = 119)


for (i in c(1:111)) {
  resultado=computeHV(as.matrix(lmknn_matrix[i,]), ref.point = c(2,2,2))
  resultados_lmknn[i]= resultado
}

print("MEJOR MODELO lmknn CONSIDERANDO HIPERVOLUMEN")
print( lmknn[which.max(resultados_lmknn),])



################################################################################
#                               Corelacion                                     #
################################################################################


correlacion_lista <- matrix(ncol = 0, nrow = 61)

for(i in c(1:61)){
  resultado <- cor(df[row.names(varImp(rfe_result))[i]], df$GRD)
  correlacion_lista[i] = resultado
  }

