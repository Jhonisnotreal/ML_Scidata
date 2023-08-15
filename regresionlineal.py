import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from siuba import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics

ruta = 'https://raw.githubusercontent.com/scidatmath2020/ML_Py_23/main/data/datos_regresion.csv'
tabla = pd.read_csv(ruta)

# print(tabla.columns)

## Graficando la dispersion ##
x = tabla["caracteristica_1"]
y = tabla["valor_real"]

ax = plt.axes()
ax.scatter(x, y, color = "black")
plt.xlabel("valor_real")
plt.ylabel("caracteristica_1")
#plt.show()

## ##

var_independientes = tabla >> select(_.caracteristica_1)  #con siuba elijo la col: caracteristica_1
objetivo = tabla >> select(_.valor_real) 
'''
print(var_independientes)
print('\n')
print(objetivo)
'''

# print(var_independientes.shape)
# print(objetivo.shape)

modelo = LinearRegression()
modelo.fit(X=var_independientes, y=objetivo) #calculando la regresion
modelo.intercept_ #Esta es la alpha de la regresion
modelo.coef_ #La beta
modelo.predict(var_independientes)
tabla["predicciones"] = modelo.predict(var_independientes)


# Calculo de errores
metrics.mean_absolute_error(tabla['valor_real'], tabla['predicciones']) #Error medio
metrics.mean_squared_error(tabla['valor_real'], tabla['predicciones']) #Erorr cuadratico medio
np.sqrt(metrics.mean_squared_error(tabla['valor_real'], tabla['predicciones']))

tabla >> mutate(error = _.valor_real-_.predicciones)

n = tabla.shape[0]
k = var_independientes.shape[1]

#Medimos la complejidad del algoritmo
R2 = metrics.r2_score(tabla["valor_real"], tabla["predicciones"])

1-(1-R2)*(n-1)/(n-k-1)
# 1-(1-R2)*(50-1)/(50-1-1)
