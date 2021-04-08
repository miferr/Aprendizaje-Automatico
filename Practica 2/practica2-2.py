#   Librerias
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


#   Convertimos el .csv en un array de numpy
valores = read_csv("ex2data2.csv", header=None).to_numpy().astype(float)

#   Separamos las coordenadas en dos arrays, X e Y
X = valores[:, :-1]
Y = valores[:, -1]

#   Creamos los nuevos atributos de cada ejemplo de entrenamiento
poly = PolynomialFeatures(6).fit_transform(X)

#   Obtenemos el numero de ejemplos de entrenamiento (m) y de atributos (n)
m = np.shape(poly)[0]
n = np.shape(poly)[1]

#   Definimos nuestra theta
theta = [0] * (n)

#   Definimoos nuestra lamda
lamda = 1

#   Definimos la funcion sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#   Defininmos la funcion de hipotesis
def hip(X, theta):
    return sigmoid(np.matmul(X, theta))

#   Definimos la funcion de coste
def cost(theta, X, Y):
    return (-1 / m) * (np.dot(Y, np.log(hip(X, theta))) + np.dot((1 - Y), np.log(1 - hip(X, theta)))) + (lamda / (2 * m)) * np.sum(np.power(theta, 2))               

#   Definimos la funcion de gradiente
def gradient(theta, X, Y):
    gradiente =  ((1 / m) * np.matmul((hip(X, theta) - Y), X))
    result = gradiente[0]
    i = 1
    for e in gradiente[1:]:
        result = np.append(result, e + ((lamda / m) * theta[i]))
        i += 1
    return result

#   Obtenemos el coste y los valores de theta optimos
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(poly, Y))
theta_opt = result[0]

#   Definimos la funcion del borde de decision a dibujar
def plot_decisionboundary(X, Y, theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max)) 
    h = sigmoid(PolynomialFeatures(6).fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
 
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig("boundary.pdf")

#   Definimos los puntos a dibujar
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
pos = np.where(Y == 1.0)
neg = np.where(Y == 0.0)

#   Dibujamos la grafica
plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='g')
plot_decisionboundary(X, Y, theta_opt, poly)

#print(cost(theta, poly, Y))
