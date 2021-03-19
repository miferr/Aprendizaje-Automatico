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

#   Funcion de normalizacion
def normalize(X):
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

# Definicion de variables
theta = [0] * (columnas + 1)
alpha = 0.1
X_norm, mu, sigma = normalize(X)
X_norm = np.hstack([np.ones([filas,1]),X_norm])
X = np.hstack([np.ones([filas,1]),X])

#   Funcion de coste
def fun_coste(X,Y,theta):
    hip = np.dot(X, theta)
    aux = (hip - Y) **2
    return aux.sum() / (2 * filas)

# Metodo de descenso de gradiente vectorizado
def gradiente(X, Y, theta, alpha):
    th = theta
    for i in range(columnas):
        aux = ((np.dot(X, theta) - Y) * X[:, i])
        th[i] -= ((alpha/filas) * aux.sum())
    return th

#θ= (XTX)−1XT~y   
def ecuacion_normal(X, Y):
    
    th = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    return th

thg = gradiente(X_norm, Y, theta, alpha)
the = ecuacion_normal(X, Y)
metros = 1650
habitaciones = 3
metros_normalizados = (metros - mu[0]) / sigma[0]
habitaciones_normalizadss = (habitaciones - mu[1]) / sigma[1]
precio1 = thg[0] + thg[1]*metros_normalizados + thg[2]*habitaciones_normalizadss
precio2 = the[0] + the[1]*metros + the[2]*habitaciones
print(precio1, precio2)




