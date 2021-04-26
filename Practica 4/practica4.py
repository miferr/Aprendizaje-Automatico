#   Librerias
from scipy.io import loadmat
import numpy as np

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
lamda = 1


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
def cost(H, Y, Theta1, Theta2, lamda):
    th1 = np.delete(Theta1, 0, axis=1)
    th2 = np.delete(Theta2, 0, axis=1)
    suma = 0
    for i in range(m):
        suma += np.sum((np.matmul(-Y[i,:], np.log(H[i,:])) - np.matmul((1 - Y[i,:]), np.log(1 - H[i,:]))))
    return ((1 / m) * suma) + ((lamda / (2 * m)) * (np.sum(np.power(th1, 2)) + np.sum(np.power(th2, 2))))

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
    
    return

A1, A2, H= forward_propagate(X, theta1, theta2)

print(cost(H, y_onehot, theta1, theta2, lamda))
