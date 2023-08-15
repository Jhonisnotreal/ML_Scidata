import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from siuba import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics

tabla = pd.read_csv('https://raw.githubusercontent.com/scidatmath2020/ML_Py_23/main/data/casas_boston.csv')
# print(tabla.columns)

var_independientes = tabla >> select(-_.MEDV, -_.RAD) #no quiero las col: MEDV y RAD
objetivo = tabla >> select(_.MEDV) #MEDV es la UNICA col que quiero

## VEO LAS COLUMNAS QUE TIENEN:  ##
#print(objetivo.shape)
#print(var_independientes.shape)

modelo_regresion = LinearRegression()
modelo_regresion.fit(X=var_independientes, y=objetivo)

tabla = tabla >> mutate(predicciones = modelo_regresion.predict(var_independientes))
