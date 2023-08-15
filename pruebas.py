import os
import pandas as pd
import numpy as np
from siuba import *
from plotnine import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics

ruta = 'datos_regresion.csv'
tabla = pd.read_csv(ruta)

print(tabla)

(ggplot(data = tabla) +
	geom_point(mapping=aes(x="caracteristica_1", y="valor_real"), color="red")
)

var_independientes = tabla >> select(_.caracteristica_1)
objetivo = tabla >> select(_.valor_real)

modelo = LinearRegression()

modelo.fit(X=var_independientes, y=objetivo)
modelo.intercept_
modelo.coef_

tabla["predicciones"] = modelo.predict(var_independientes)

(ggplot(data = tabla) +
	geom_point(mapping=aes(x="caracteristica_1", y="valor_real"), color="blue") +
	geom_point(mapping=aes(x="caracteristica_1", y="predicciones"), color="red") +
	geom_abline(slope=1.85, intercept=5.711) +
	geom_smooth(mapping=aes(x="caracteristica_1", y="valor_real"), color="green")
)

modelo.coef_
modelo.intercept_

metrics.mean_absolute_error(tabla["valor_real"], tabla["predicciones"])
metrics.mean_squared_error(tabla["valor_real"], tabla["predicciones"])
np.sqrt(metrics.mean_squared_error(tabla["valor_real"], tabla["predicciones"]))

tabla >> mutate(error = _.valor_real-_.predicciones)

R2 = metrics.r2_score(tabla["valor_real"], tabla["predicciones"])

1-(1-R2)*(50-1)/(50-1-1)

n = tabla.shape[0]
k = var_independientes.shape[1]

1-(1-R2)*(n-1)/(n-k-1)




##
ruta = 'casas_boston.csv'
tabla = pd.read_csv(ruta)

var_independientes = tabla >> select(-_.MEDV, -_.RAD)
objetivo = tabla >> select(_.MEDV)

modelo_regresion = LinearRegression()
modelo_regresion.fit(X=var_independientes, y=objetivo)

tabla = tabla >> mutate(predicciones = modelo_regresion.predict(var_independientes)) >> select(-_.RAD)
# tabla["predicciones"] = modelo_regresion.predict(var_independientes) >> select(-_.RAD)

