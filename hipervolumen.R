library("ecr")
names <- c("rf","svr","knn","kknn","gbm","lm_rf","lm_knn")

# Variables vacías
resultados_hipervolumen <- data.frame(rf = numeric(),
                      svr = numeric(),
                      knn = numeric(),
                      kknn = numeric(),
                      gbm= numeric(),
                      lm_rf  = numeric(),
                      lm_knn= numeric())

#######################################################################

vector_2014=rbind(
                  c(0.5076,1.0000,1.0000),
                  c(0.7032,0.6893,0.8674),
                  c(0.6968,0.6852,0.4465),
                  c(0.0000,0.0000,0.0000),
                  c(0.4506,0.5054,0.6710),
                  c(1.0000,0.8577,0.9587),
                  c(0.7101,0.6937,0.9934),
                  c(0.8121,0.7560,0.7477)
                  )

resultados_2014= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_2014[i,]), ref.point = c(2,2,2))
  resultados_2014[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_2014)

#######################################################################

vector_2015=rbind(
                  c(0.0000, 1.0000, 1.0000),
                  c(0.1707, 0.1170, 0.2543),
                  c(0.5833, 0.4064, 0.5376),
                  c(0.5540, 0.3908, 0.3919),
                  c(0.0609, 0.0000, 0.0000),
                  c(1.0000, 0.5805, 0.2241),
                  c(0.2296, 0.1703, 0.2594),
                  c(0.4435, 0.3267, 0.0188)
                  
)
resultados_2015= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_2015[i,]), ref.point = c(2,2,2))
  resultados_2015[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_2015)

#########################################################################


vector_2016=rbind(
                  c(0.0019, 0.8007, 1.0000),
                  c(0.1293, 0.3180, 0.3152),
                  c(0.0000, 0.0000, 0.0000),
                  c(0.0521, 0.1679, 0.0616),
                  c(0.3886, 0.5442, 0.4938),
                  c(0.1561, 0.3556, 0.6336),
                  c(0.8505, 0.6807, 0.5649),
                  c(1.0000, 1.0000, 0.8301)
                  
                  
)
resultados_2016= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_2016[i,]), ref.point = c(2,2,2))
  resultados_2016[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_2016)


#########################################################################

vector_2017=rbind(
                  c(0.0505, 0.5941, 0.9310),
                  c(0.0425, 0.0948, 0.0362),
                  c(0.0000, 0.0000, 0.0275),
                  c(0.2562, 0.4367, 0.3029),
                  c(1.0000, 1.0000, 1.0000),
                  c(0.1641, 0.3099, 0.0000),
                  c(0.9711, 0.9862, 0.7073),
                  c(0.3798, 0.5755, 0.7501)
)
resultados_2017= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_2017[i,]), ref.point = c(2,2,2))
  resultados_2017[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_2017)

#########################################################################


vector_2018=rbind(
                  c(0.2227, 1.0000, 1.0000),
                  c(0.0000, 0.0000, 0.0000),
                  c(0.0772, 0.1411, 0.0510),
                  c(0.2239, 0.3338, 0.2244),
                  c(0.3284, 0.4359, 0.3978),
                  c(0.2949, 0.4055, 0.2764),
                  c(1.0000, 0.8025, 0.5642),
                  c(0.3597, 0.4626, 0.4522)
                  
                  
)
resultados_2018= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_2018[i,]), ref.point = c(2,2,2))
  resultados_2018[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_2018)

#########################################################################

vector_2019=rbind(
                  c(0.5558, 0.0000, 0.2427),
                  c(0.8928, 0.9344, 0.5720),
                  c(1.0000, 1.0000, 1.0000),
                  c(0.7354, 0.8298, 0.4946),
                  c(0.8717, 0.9209, 0.7583),
                  c(0.7376, 0.8313, 0.5787),
                  c(0.0000, 0.1389, 0.0000),
                  c(0.3887, 0.5555, 0.7255)
                  
                  
)
resultados_2019= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_2019[i,]), ref.point = c(2,2,2))
  resultados_2019[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_2019)

#########################################################################

vector_2020=rbind(
                  c(0.0599, 0.7724, 0.9812),
                  c(0.1904, 0.3730, 0.6016),
                  c(0.0000, 0.0000, 0.0000),
                  c(0.0577, 0.1904, 0.3270),
                  c(0.0269, 0.1079, 0.2547),
                  c(0.8666, 0.5833, 0.9422),
                  c(1.0000, 1.0000, 1.0000),
                  c(0.2125, 0.3911, 0.5645)
                  
                  
)
resultados_2020= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_2020[i,]), ref.point = c(2,2,2))
  resultados_2020[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_2020)



#########################################################################


vector_general=rbind(
                  c(1.0000, 1.0000, 1.0000),
                  c(0.0546, 0.1074, 0.0729),
                  c(0.0000, 0.0000, 0.0000),
                  c(0.0051, 0.0102, 0.1418),
                  c(0.1519, 0.2878, 0.0649),
                  c(0.5279, 0.8830, 0.4457),
                  c(0.1784, 0.3347, 0.1150),
                  c(0.1562, 0.2954, 0.1203)
                  

                  
)
resultados_general= matrix(ncol = 0, nrow = 8)

for (i in c(1:8)) {
  resultado=computeHV(as.matrix( vector_general[i,]), ref.point = c(2,2,2))
  resultados_general[i]= resultado
}
resultados_hipervolumen = rbind(resultados_hipervolumen, resultados_general)
##################################################################################


colnames(resultados_hipervolumen) <- c("ML", "rf","svr","knn","kknn","gbm","lm_rf","lm_knn")

rownames(resultados_hipervolumen) <- c("2014", "2015", "2016", "2017", "2018", "2019", "2020", "GENERAL")

write.csv(t(resultados_hipervolumen), file = "result_hipervolumen.csv", row.names = TRUE)

##############################################################################


