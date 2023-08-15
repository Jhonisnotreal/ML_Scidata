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



