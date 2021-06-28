import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Cargamos los datos
datos = pd.read_csv("water_potability.csv")
datos.head()

datos.isnull().sum()

datos = datos.dropna()
datos.Potability.value_counts().plot(kind='pie')

zero  = datos[datos['Potability']==0]  
one = datos[datos['Potability']==1] 

df_minority_upsampled = resample(one, replace = True, n_samples = 1200) 

datos = pd.concat([zero, df_minority_upsampled])
datos = shuffle(datos) 

datos.Potability.value_counts().plot(kind='pie')

X = datos.drop(['Potability'], axis = 1)
y = datos['Potability']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features= X.columns
X[features] = sc.fit_transform(X[features])



x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)

ox_train = np.hstack([np.ones([np.shape(x_train)[0],1]),x_train])
ox_test = np.hstack([np.ones([np.shape(x_test)[0],1]),x_test])

y_onehot = np.zeros((np.shape(x_train)[0], 2))
for i in range(np.shape(x_train)[0]):
    y_onehot[i][y_train.iloc[i]] = 1.0

#-------------------------------------------------------
#   Definimos la funcion sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#   Defininmos la funcion de hipotesis
def hip(a, b):
    return sigmoid(np.matmul(a, b))

#   Definimos la funcion de propagacion hacia adelante
def forward_propagate(X, Theta1, Theta2):
    m = X.shape[0]
    A1 = np.hstack([np.ones([m, 1]), X])
    Z2 = np.dot(A1, Theta1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoid(Z2)])
    Z3 = np.dot(A2, Theta2.T)
    H = sigmoid(Z3)
    return A1, Z2, A2, Z3, H

def cost(H, Y, Theta1, Theta2, reg):
    m = len(Y)
    th1 = np.delete(Theta1, 0, axis=1)
    th2 = np.delete(Theta2, 0, axis=1)
    suma = 0
    for i in range(m):
        suma += np.sum((np.matmul(-Y[i,:], np.log(H[i,:])) - np.matmul((1 - Y[i,:]), np.log(1 - H[i,:]))))
    return ((1 / m) * suma) + ((reg / (2 * m)) * (np.sum(np.power(th1, 2)) + np.sum(np.power(th2, 2))))


def cash_converter(H):
    x = []
    for e in H:
        if e[0] > e[1]:
            x.append(0)
        else:
            x.append(1)
    return x

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    m = X.shape[0]
    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    A1, Z2, A2, Z3, H = forward_propagate(X, Theta1, Theta2)
    #Llamada a la funcion para calcular el coste

    coste = cost(H, y, Theta1, Theta2, reg)

    #Back-propagation
    Delta1, Delta2 = np.zeros(Theta1.shape), np.zeros(Theta2.shape)
    sigma3 = (H - y)
    Delta2 += np.dot(sigma3.T, A2)
    Delta1 += np.dot(np.delete(np.dot(sigma3, Theta2) * (A2 * (1 - A2)), 0, axis=1).T, A1)
    D1 = Delta1 / m
    D2 = Delta2 / m
    #Regularizacion del gradiente
    D1[:, 1:] = D1[:, 1:] + (reg * Theta1[:, 1:]) / m
    D2[:, 1:] = D2[:, 1:] + (reg * Theta2[:, 1:]) / m
       
    return coste, np.concatenate((D1, D2), axis=None)


def pesosAleatorios(L_in, L_out):
    ini_epsilon = 0.12
    theta = np.random.rand(L_out, 1 + L_in) * (2*ini_epsilon) - ini_epsilon 
    return theta 

def train(X, y, reg, iters):
    num_entradas = X.shape[1]
    num_ocultas = 64
    num_etiquetas = 2

    theta1 = pesosAleatorios(num_entradas, num_ocultas)
    theta2 = pesosAleatorios(num_ocultas, num_etiquetas)
    params = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

    fmin = opt.minimize(fun=backprop, x0=params, 
                 args=(num_entradas, num_ocultas, num_etiquetas, X, y, reg),
                 method='TNC', jac=True, options={'maxiter' : iters})

    theta1 = np.reshape(fmin.x[:num_ocultas * (num_entradas + 1)],
                       (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(fmin.x[num_ocultas * (num_entradas + 1):],
                       (num_etiquetas, (num_ocultas + 1)))

    a1, z2, a2, z2, h = forward_propagate(X, theta1, theta2)

    predictions = np.argmax(h, axis=1)
    return predictions



pre = train(x_train, y_onehot, 1, 70)
print(pre)