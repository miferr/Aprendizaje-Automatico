#   Librerias
from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize


#   Obtenemos los datos del dataset
data = loadmat("ex5data1.mat")
X = data["X"]
y = data["y"].ravel()
Xtest = data["Xtest"]
ytest = data["ytest"].ravel()
Xval = data["Xval"]
yval = data["yval"].ravel()

#   Obtenemos el numero de ejemplos de entrenamiento (m) y de atributos (n)
m = len(y)
n = np.shape(X)[1]

#   Añadimos la columna de ceros a X
OX = np.hstack([np.ones([m,1]),X])

#   REGRESION LINEAL REGULARIZADA
#   Definimos nuestro termino de regulacion
reg = 0

#   Definimos nuestra theta
theta = np.array([1] * (n + 1))

#   Defininmos la funcion de hipotesis
def hip(X, theta):
    return np.dot(X, theta)

#   Definimos la funcion de coste
def cost(theta, X, Y, reg):
    return (1 / (2 * m)) * np.sum((hip(X, theta) - Y) ** 2) + ((reg / (2 * m)) * np.sum(theta[1:] ** 2))          

#   Definimos la funcion de gradiente
def gradient(theta, X, Y, reg):
    gradiente = (1 / m) * np.dot((hip(X, theta) - Y), X)
    result = np.zeros(np.shape(X)[1])
    result[0] = gradiente[0]
    result[1:] = gradiente[1:] + (theta[1:] * (reg / m))
    return  result

#   Definimos la funcion global de regresion lineal vectorizada
def reg_lineal_vec(X, Y, theta, reg):
    return cost(X, Y, theta, reg), gradient(X, Y, theta, reg)

print(np.ones(np.shape(X)[1]))
print(gradient(theta, OX, y, reg))
fmin = minimize(fun=cost, x0=theta, args=(OX, y, reg), jac=gradient)
theta_opt  = fmin.x
print(fmin)

