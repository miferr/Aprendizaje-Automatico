# Practica 1 - Parte 2

#   Librerias
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from matplotlib import cm


#   Convertimos el .csv en un array de numpy
valores = read_csv("ex1data2.csv", header=None).to_numpy().astype(float)

#   Separamos las coordenadas en dos arrays, X e Y
X = valores[:, :-1]
Y = valores[:, -1]

#   Obtenemos las dimensiones de la matriz X (n x m)
filas = np.shape(X)[0]
columnas = np.shape(X)[1]

# Definicion de variables
theta = [0] * (columnas + 1)
alpha = 0.01

#   Funcion de normalizacion
def normalize():
    mu = []
    des = []
    x_norm = np.empty_like(X)
    for i in range(columnas):
        #   Calculamos la media y la desviacion
        mu.append(np.mean(X[:,i]))
        des.append(np.std(X[:,i]))
        #   Calculamos el valor normalizado
        for k in range(filas):
            x_norm[k][i] = (X[k][i] - mu[i]) / des[i]
    return x_norm, mu, des

#   Funcion de coste
def fun_coste():
    hip = np.dot(X, theta)
    aux = (hip - Y) **2
    return aux.sum() / (2 * filas)

print(theta)
# Metodo de descenso de gradiente vectorizado
norm,med,des = normalize()
X = np.hstack([np.ones([filas,1]),X])
for i in range(10):
    th = theta
    for i in range(columnas):
        print(str((np.dot(X, theta) - Y) * X[:, i]))
        #th[i] -= ((alpha/filas) * aux.sum())
    

print(theta)

