#   Librerias
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


#   Extraemos los ejemplos de entrenamiento del archivo que los contiene
data = loadmat("ex3data1.mat")

#   Separamos los atributos de los ejemplos y sus etiquetas (El 0 viene etiquetado como 10)
y = data['y']
X = data['X']

#   Obtenemos las etiquetas unicas
etiquetas = np.unique(y)

#   Obtenemos el numero de ejemplos de entrenamiento, sus numeros de pixeles (m) y las etiquetas
num_ejemplos = np.shape(X)[0]
m = np.shape(X)[1]
num_etiquetas = etiquetas.size

#   Añadimos una columna de unos en X para facilitar los calculos con matrices
OX = np.hstack([np.ones([num_ejemplos, 1]), X])

#   Definimos nuestra theta
theta = np.zeros(m +  1)

#   Definimoos nuestra termino de regularizacion
lamda = 0.1

# Seleccionamos aleatoriamente 10 ejemplos y los pintamos 
sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample, :].reshape(-1, 20).T)
plt.axis('off')

#   Definimos la funcion sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#   Defininmos la funcion de hipotesis
def hip(X, theta):
    return sigmoid(np.matmul(X, theta))

#   Definimos la funcion de coste
def cost(theta, X, Y):
        return (-1 / m) * (np.dot(np.ravel(Y), np.log(hip(X, theta))) + np.dot((1 - np.ravel(Y)), np.log(1 - hip(X, theta) + 1e6))) + (lamda / (2 * m)) * np.sum(np.power(theta, 2))       

#   Definimos la funcion de gradiente
def gradient(theta, X, Y):
    return (1 / m) * np.matmul((hip(X, theta) - np.ravel(Y)), X)

#   Definimos la función de entrenamiento para los clasificadores
def oneVsAll(X, y, etiquetas, reg):
    result = []
    for i in etiquetas:
        aux = np.asarray(y == i) * 1
        op = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(OX, aux), disp=False)

        result.append(op[0])
    return result

#   Obtenemos los valores optimos de theta
theta_opt = oneVsAll(X, y, etiquetas, lamda)

#   Definimos una funcion  que calcule el porcentaje de predicciones correctas
def porcentaje(theta):
    ok = 0
    i = 0
    for h in hip(OX, theta):
        print(h)
        i +=1
    return (ok / m)

porcentaje(theta_opt[9])
