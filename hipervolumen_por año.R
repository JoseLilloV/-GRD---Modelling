library(ecr)
library(dplyr)
library(caret)

rf <- read.csv( "result_rf.txt")
rf <- select(rf, n, R2, RMSE, MAE)
rf$R2[rf$R2<0.001] <- 0.001

rf$R2 <- 1/rf$R2

preproc <- preProcess(select(rf, -n), method = c("range"))
rf <- predict(preproc, rf)
rf_matrix <- as.matrix(select(rf, -n))
resultados_rf = matrix(ncol = 0, nrow = 119)


for (i in c(1:112)) {
  resultado=computeHV(as.matrix(rf_matrix[i,]), ref.point = c(2,2,2))
  resultados_rf[i]= resultado
}

print("MEJOR MODELO RF CONSIDERANDO HIPERVOLUMEN")
print( rf[which.max(resultados_rf),])


#################################################################
svr <- read.csv( "result_svr.txt")
svr <- select(svr, n, R2, RMSE, MAE)
svr$R2[svr$R2<0.001] <- 0.001

svr$R2 <- 1/svr$R2

preproc <- preProcess(select(svr, -n), method = c("range"))
svr <- predict(preproc, svr)
svr_matrix <- as.matrix(select(svr, -n))
resultados_svr= matrix(ncol = 0, nrow = 119)


for (i in c(1:112)) {
  resultado=computeHV(as.matrix(svr_matrix[i,]), ref.point = c(2,2,2))
  resultados_svr[i]= resultado
}

print("MEJOR MODELO SVR CONSIDERANDO HIPERVOLUMEN")
print( svr[which.max(resultados_svr),])

#################################################################
gbm <- read.csv( "result_gbm.txt")
gbm <- select(gbm, n, R2, RMSE, MAE)
gbm$R2[gbm$R2<0.001] <- 0.001

gbm$R2 <- 1/gbm$R2

preproc <- preProcess(select(gbm, -n), method = c("range"))
gbm <- predict(preproc, gbm)
gbm_matrix <- as.matrix(select(gbm, -n))
resultados_gbm= matrix(ncol = 0, nrow = 119)


for (i in c(1:112)) {
  resultado=computeHV(as.matrix(gbm_matrix[i,]), ref.point = c(2,2,2))
  resultados_gbm[i]= resultado
}

print("MEJOR MODELO GBM CONSIDERANDO HIPERVOLUMEN")
print( gbm[which.max(resultados_gbm),])

#################################################################
knn <- read.csv( "result_knn.txt")
knn <- select(knn, n, R2, RMSE, MAE)
knn$R2[knn$R2<0.001] <- 0.001

knn$R2 <- 1/knn$R2

preproc <- preProcess(select(knn, -n), method = c("range"))
knn <- predict(preproc, knn)
knn_matrix <- as.matrix(select(knn, -n))
resultados_knn= matrix(ncol = 0, nrow = 119)


for (i in c(1:112)) {
  resultado=computeHV(as.matrix(knn_matrix[i,]), ref.point = c(2,2,2))
  resultados_knn[i]= resultado
}

print("MEJOR MODELO KNN CONSIDERANDO HIPERVOLUMEN")
print( knn[which.max(resultados_knn),])
#################################################################
kknn <- read.csv( "result_kknn.txt")
kknn <- select(kknn, n, R2, RMSE, MAE)
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

#################################################################
lmrf <- read.csv( "result_stacking_lm_rf.txt")
lmrf <- select(lmrf, n, R2, RMSE, MAE)
lmrf$R2[lmrf$R2<0.001] <- 0.001
lmrf$R2 <- 1/lmrf$R2

preproc <- preProcess(select(lmrf, -n), method = c("range"))
lmrf <- predict(preproc, lmrf)
lmrf_matrix <- as.matrix(select(lmrf, -n))
resultados_lmrf= matrix(ncol = 0, nrow = 119)


for (i in c(1:104)) {
  resultado=computeHV(as.matrix(lmrf_matrix[i,]), ref.point = c(2,2,2))
  resultados_lmrf[i]= resultado
}

print("MEJOR MODELO lmrf CONSIDERANDO HIPERVOLUMEN")
print( lmrf[which.max(resultados_lmrf),])

#################################################################
lmknn <- read.csv( "result_stacking_lm_knn.txt")
lmknn <- select(lmknn, n, R2, RMSE, MAE)

lmknn$R2[lmknn$R2<0.001] <- 0.001
lmknn$R2 <- 1/lmknn$R2

preproc <- preProcess(select(lmknn, -n), method = c("range"))
lmknn <- predict(preproc, lmknn)
lmknn_matrix <- as.matrix(select(lmknn, -n))
resultados_lmknn= matrix(ncol = 0, nrow = 119)


for (i in c(1:104)) {
  resultado=computeHV(as.matrix(lmknn_matrix[i,]), ref.point = c(2,2,2))
  resultados_lmknn[i]= resultado
}

print("MEJOR MODELO lmknn CONSIDERANDO HIPERVOLUMEN")
print( lmknn[which.max(resultados_lmknn),])
