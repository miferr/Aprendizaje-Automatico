#   Librerias
from scipy.io import loadmat
import numpy as np
import checkNNGradients as ch

#   Extraemos los ejemplos de entrenamiento del archivo que los contiene
data = loadmat("ex4data1.mat")

#   Separamos los atributos de los ejemplos y sus etiquetas (El 0 viene etiquetado como 10)
y = data['y'].ravel()
X = data['X']
m = len(y)
input_size = X.shape[1]

#   Obtenemos las distintas etiquetas
etiquetas = np.unique(y)
num_labels = etiquetas.size

#   Definimos nuestra termino de regularizacion
reg = 1


#   Creamos la salida de y como onehot
y = (y - 1)
y_onehot = np.zeros((m, num_labels))    #   5000 x 10
for i in range(m):
    y_onehot[i][y[i]] = 1

#   Obtenemos las matrices de pesos del archivo
weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

#   Definimos la funcion sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#   Defininmos la funcion de hipotesis
def hip(a, b):
    return sigmoid(np.matmul(a, b))

#   Definimos funcion de coste
def cost(H, Y, Theta1, Theta2, reg):
    th1 = np.delete(Theta1, 0, axis=1)
    th2 = np.delete(Theta2, 0, axis=1)
    suma = 0
    for i in range(m):
        suma += np.sum((np.matmul(-Y[i,:], np.log(H[i,:])) - np.matmul((1 - Y[i,:]), np.log(1 - H[i,:]))))
    return ((1 / m) * suma) + ((reg / (2 * m)) * (np.sum(np.power(th1, 2)) + np.sum(np.power(th2, 2))))

#   Definimos la funcion de propagacion hacia adelante
def forward_propagate(X, Theta1, Theta2):
    m = X.shape[0]
    A1 = np.hstack([np.ones([m, 1]), X])
    Z2 = np.dot(A1, Theta1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoid(Z2)])
    Z3 = np.dot(A2, Theta2.T)
    H = sigmoid(Z3)
    return A1, A2, H

#   Definimos la funcion de propagacion hacia 
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    A1, A2, H = forward_propagate(X, Theta1, Theta2)
    coste = cost(H, y, Theta1, Theta2, reg)
    Delta1, Delta2 = np.zeros(Theta1.shape), np.zeros(Theta2.shape)
    Error3 = (H - y)
    Delta2 += np.dot(Error3.T, A2)
    Delta1 += np.dot(np.delete(np.dot(Error3, Theta2) * (A2 * (1 - A2)), 0, axis=1).T, A1)
    D1 = Delta1 / m
    D2 = Delta2 / m
    
    return coste, np.concatenate((D1, D2), axis=None)

param = np.concatenate((theta1, theta2), axis=None)
#coste, gradiente = backprop(param, input_size, 25, num_labels, X, y_onehot, reg)

print(ch.checkNNGradients(backprop, 0))
   
