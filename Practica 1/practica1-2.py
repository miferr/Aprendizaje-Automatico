# Practica 1 - Parte 2

# Librerias
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from matplotlib import cm


#  Convertimos el .csv en un array de numpy
valores = read_csv("ex1data2.csv", header=None).to_numpy().astype(float)

#  Separamos las coordenadas en dos arrays, X e Y
X = valores[:, :-1]
Y = valores[:, -1]

#  Obtenemos las dimensiones de la matriz X (n x m)
filas = np.shape(X)[0]
columnas = np.shape(X)[1]


def normalize(X):
    '''Normaliza los datos de la matriz X
    restandoles su media y dividiendolos por la desviacion estandar'''
    mu = np.mean(X, axis=0)
    X_mean = X - mu
    sigma = np.std(X, axis=0)
    X_norm = X_mean / sigma
    return X_norm, mu, sigma


# Normalizamos los datos
theta = [0] * (columnas + 1)

valores_norm, mu, sigma = normalize(valores)
X_norm = valores_norm[:, :-1]
Y_norm = valores_norm[:, -1]


# Añadimos una columna de 1s a X
X_norm = np.hstack([np.ones([filas,1]),X_norm])

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
    return th, fun_coste(X, Y, th)

# Ecuacion normal
def ecuacion_normal(X, Y):
    X= np.hstack([np.ones([filas,1]),X])
    th = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    return th

# Pruebas con alpha
for alpha in [0.3, 0.1, 0.03, 0.01]:
    theta = np.zeros(columnas + 1)
    costes = []
    plt.figure(1)
    plt.title(alpha)
    plt.xlabel('Iteracion')
    plt.ylabel('Coste')
    
    for i in range(3000):
        Theta, cost = gradiente(X_norm, Y_norm, theta, alpha)
        costes.append(cost)
    
    plt.plot(costes)
    plt.show()
    print('Coste final:', cost)
    print('Theta final:', theta)

# Comprobacion de las funciones
# Calculamos las thetas
thg = np.zeros(columnas + 1)
alpha = 0.1

for i in range(3000):
    thg, cost = gradiente(X_norm, Y_norm, thg, alpha)

the = ecuacion_normal(X, Y)

# Creamos datos de prueba y los normalizamos
casa = np.array([1650,3]) 
casa_norm = (casa - mu[:-1])
casa_norm = casa_norm / sigma[:-1]

# Prediccion con descenso del gradiente
gradient_predict = np.matmul(np.append(np.array([1]),casa_norm), thg) * sigma[-1]
gradient_predict = gradient_predict + mu[-1]

# Prediccion con ecuación normal
normal_predict = np.matmul(np.append(np.array([1]),casa), the)

print('Precio con', casa[0], 'pies cuadrados y', casa[1], 'habitaciones:')
print('Resultado con descenso del gradiente:', gradient_predict)
print('Resultado con ecuación normal:', normal_predict)