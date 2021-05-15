#   Librerias
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#   Funcion de visualizacion de las graficas
def visualize_data(X, y, file_name):
    pos = np.where(y == 1.0)
    neg = np.where(y == 0.0)
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='g')
    plt.savefig(file_name)
    plt.close()

def visualize_boundary(X, y, svm, file_name):
    margin = 0.05
    x1 = np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, 100)
    x2 = np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = np.where(y == 1.0)
    neg = np.where(y == 0.0)
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='g')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.close()


### 1.1 SVM CON KERNEL LINEAL
#   Cargamos los datos del primer conjunto
data = loadmat("ex6data1.mat")
X = data["X"]
y = data["y"].ravel()

#   Visualizamos los datos del primer conjunto
visualize_data(X, y, "data_1")
    
#   Creamos el clasificador linal y lo entrenamos para C = 1
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)
visualize_boundary(X, y, svm, "svm_l_c_1")

#   Creamos el clasificador lineal y lo entrenamos para C = 100
svm = SVC(kernel='linear', C=100)
svm.fit(X, y)
visualize_boundary(X, y, svm, "svm_l_c_100")


### 1.2 SVM CON KERNEL GAUSSIANO
#   Cargamos los datos del segundo conjunto
data = loadmat("ex6data2.mat")
X = data["X"]
y = data["y"].ravel()

#   Definimos las variables que vamos a usar en el entrenamiento del svm
C = 1.0
sigma = 0.1

#   Visualizamos los datos del segundo conjunto
visualize_data(X, y, "data_2")

#   Creamos el clasificador gaussiano y lo entrenamos
svm = SVC(kernel='rbf', C=C, gamma=1 / (2 * sigma**2))
svm.fit(X, y)
visualize_boundary(X, y, svm, "svm_g")


### 1.3 ELECCION DE LOS PARAMETROS C Y SIGMA
#   Cargamos los datos del tercer conjunto
data = loadmat("ex6data3.mat")
X = data["X"]
y = data["y"].ravel()
Xval = data["Xval"]
yval = data["yval"].ravel()

#   Definimos nuestro conjunto de valores para C y sigma
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

#   Calculamos que combinacion es la mejor
best_C = 0
best_sigma = 0
best_acc = 0
for C in values:
    for sigma in values:
        svm = SVC(kernel='rbf', C=C, gamma=1 / (2 * sigma**2))
        svm.fit(X, y)
        acc = accuracy_score(yval, svm.predict(Xval))
        if acc >= best_acc:
            best_C = C
            best_sigma = sigma
            best_acc = acc

print("Mejor C: ", best_C, ", Mejor sigma: ", best_sigma, ", Mejor precision: ", best_acc)
svm = SVC(kernel='rbf', C=best_C, gamma=1 / (2 * best_sigma**2))
svm.fit(X, y)
visualize_boundary(X, y, svm, "svm_g_best")
