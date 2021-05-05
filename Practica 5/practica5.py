#   Librerias
from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#   Obtenemos los datos del dataset
data = loadmat("ex5data1.mat")
X = data["X"]
y = data["y"].ravel()
Xtest = data["Xtest"]
ytest = data["ytest"].ravel()
Xval = data["Xval"]
yval = data["yval"].ravel()
#   Obtenemos el numero de ejemplos de entrenamiento (m) y de atributos (n)
m = np.shape(X)[0]
n = np.shape(X)[1]

#   Añadimos la columna de ceros a X
OX = np.hstack([np.ones([m,1]),X])
Xval = np.hstack([np.ones([np.shape(Xval)[0], 1]), Xval])
Xtest = np.hstack([np.ones([np.shape(Xtest)[0], 1]), Xtest])

#   REGRESION LINEAL REGULARIZADA

#   Definimos nuestra theta
theta = np.array([1] * (n + 1))

#   Defininmos la funcion de hipotesis
def hip(X, theta):
    return np.dot(X, theta)

def error(theta, X, Y):
    m = np.shape(X)[0]
    return (1 / (2 * m)) * np.sum(((hip(X, theta) - Y) ** 2))

#   Definimos la funcion de coste
def cost(theta, X, Y, reg):   
    return error(theta,X,Y) + ((reg / (2 * m)) * np.sum(theta[1:] ** 2))          

#   Definimos la funcion de gradiente
def gradient(theta, X, Y, reg):
    gradiente = (1 / m) * np.dot((hip(X, theta) - Y), X)
    result = np.zeros(np.shape(X)[1])
    result[0] = gradiente[0]
    result[1:] = gradiente[1:] + (theta[1:] * (reg / m))
    return  result

def learning_curve(X, Y, Xval, yval,reg):
    cross_validation_error = []
    train_error = []
    n_training_sets = []
    theta = np.array([1] * (n + 1))
    for i in range (1, m):
        theta_opt = minimize(fun=cost, x0=theta, args=(OX[:i], y[:i], reg), jac=gradient).x
        cross_validation_error.append(error(theta_opt,Xval,yval))
        train_error.append(error(theta_opt,OX[:i],y[:i]))
        n_training_sets.append(i)
    return n_training_sets, train_error, cross_validation_error

#   Definimos la funcion global de regresion lineal vectorizada
def reg_lineal_vec(X, Y, theta, reg):
    return cost(X, Y, theta, reg), gradient(X, Y, theta, reg)

def grafica_regresion_lineal(X,Y,theta): 
    plt.plot(X,Y, "x", color="red")
    min_x = min(X)
    max_x = max(X)
    min_y = theta[0] + (theta[1]*min_x)
    max_y = theta[0] + (theta[1]*max_x)
    plt.plot([min_x, max_x], [min_y, max_y], color="blue")
    
def grafica_learning_curve(n_training_sets,train_error,cross_validation_error):
    plt.plot(n_training_sets,train_error,label='Training error')
    plt.plot(n_training_sets,cross_validation_error,label='Cross Validation error')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
    
   #EJECUCION---------------------------------------------
reg = 1
theta_opt = minimize(fun=cost, x0=theta, args=(OX, y, reg), jac=gradient).x
plt.figure(1)
grafica_regresion_lineal(X,y,theta_opt)
#-----------------------------------------------------------------------
reg = 0
n_training_sets, train_error, cross_validation_error = learning_curve(OX, y, Xval, yval, reg)
plt.figure(2)
grafica_learning_curve(n_training_sets, train_error, cross_validation_error)


