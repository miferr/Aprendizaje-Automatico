#   Librerias
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt


#   Convertimos el .csv en un array de numpy
valores = read_csv("ex2data1.csv", header=None).to_numpy().astype(float)

#   Separamos las coordenadas en dos arrays, X e Y
X = valores[:, :-1]
Y = valores[:, -1]

#   Obtenemos el numero de ejemplos de entrenamiento (m) y de atributos (n)
m = np.shape(X)[0]
n = np.shape(X)[1]

#   Añadimos una columna de unos en X para facilitar los calculos con matrices
OX = np.hstack([np.ones([m, 1]), X])

#   Definimos nuestra theta
theta = [0] * (n + 1)

#   Definimos la funcion sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#   Defininmos la funcion de hipotesis
def hip(X, theta):
    return sigmoid(np.matmul(X, theta))

#   Definimos la funcion de coste
def cost(theta, X, Y):
    return (-1 / m) * (np.dot(Y, np.log(hip(X, theta))) + np.dot((1 - Y), np.log(1 - hip(X, theta))))

#   Definimos la funcion de gradiente
def gradient(theta, X, Y):
    return (1 / m) * np.matmul((hip(X, theta) - Y), X)

#   Obtenemos el coste y los valores de theta optimos
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(OX, Y))
theta_opt = result[0]

#   Definimos una funcion  que calcule el porcentaje de predicciones correctas
def porcentaje():
    ok = 0
    i = 0
    for h in hip(OX, theta_opt):
        if h >= 0.5:
            if Y[i] == 1.0:
                ok +=1
        else:
            if Y[i] == 0.0:
                ok +=1
        i +=1
    return (ok / m)

#   Definimos la funcion de la recta a dibujar
def pinta_frontera_recta(X, Y, theta):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta)) 
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.pdf")

#   Definimos los puntos a dibujar
plt.xlabel('Puntuacion Examen 1')
plt.ylabel('Puntuacion Examen 2')
pos = np.where(Y == 1.0)
neg = np.where(Y == 0.0)

#   Dibujamos la grafica
plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='g')
pinta_frontera_recta(X, Y, theta_opt)

#   Calculamos el porcentaje de aciertos del modelo
print("Porcentaje de aciertos: " + str(porcentaje()))
