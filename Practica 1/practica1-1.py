#   Practica 1 - Parte 1


#   Librerias
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from matplotlib import cm

#   Declaracion de las variables
theta = [0,0]
alpha = 0.01
iteraciones = 1500
rango_th0 = [-10,10]
rango_th1 = [-1,4]

#   Convertimos el .csv en un array de numpy
valores = read_csv("ex1data1.csv", header=None).to_numpy()

#   Separamos las coordenadas en dos arrays, X e Y
X = valores[:,0]
Y = valores[:,1]

#   Obtenemos el numero de elementos m
m = len(X)

#   Funcion de hipotesis
def fun_hip(x):
    return theta[0] + (theta[1]*x)

#   Funcion de coste
def fun_coste(th):
    sum = 0;
    for i in range(m):
        sum += ((th[0] + th[1] * X[i]) - Y[i])**2
    return sum /(m*2)

#   Funcion para obtener los datos de la grafica 3D
def make_data():
    step = 0.1
    th0 = np.arange(rango_th0[0], rango_th0[1], step)
    th1 = np.arange(rango_th1[0], rango_th1[1], step)
    th0, th1 = np.meshgrid(th0, th1)
    coste = np.empty_like(th0)
    for ix, iy in np.ndindex(th0.shape):
        coste[ix, iy] = fun_coste([th0[ix, iy], th1[ix, iy]])
    return [th0, th1, coste]

#   Metodo de descenso de gradiente
for i in range(iteraciones):
    sum0 = sum1 = 0
    for i in range(m):
        sum0 += fun_hip(X[i]) - Y[i]
        sum1 += (fun_hip(X[i]) - Y[i]) * X[i]
    theta[0] -= (alpha / m) * sum0
    theta[1] -= (alpha / m) * sum1

#   Dibujamos la grafica 2D
# Dibujamos los puntos [x,y] del .csv
plt.plot(X,Y, "x")
# Dibujamos la recta de la funcion de hipotesis 
min_x = min(X)
max_x = max(X)
min_y = fun_hip(min_x)
max_y = fun_hip(max_x)
plt.plot([min_x, max_x], [min_y, max_y])
plt.savefig("resultado.pdf")

#   Dibujamos la grafica 3D
eje_x, eje_y, eje_z = make_data()
cos = plt.figure()
ejes =  cos.gca(projection = "3d")
surface = ejes.plot_surface(eje_x, eje_y, eje_z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("coste.pdf")

#   Dibujamos la grafica de contorno
con = plt.figure()
plt.plot(theta[0], theta[1], "x")
plt.contour(eje_x, eje_y, eje_z,np.logspace(-2, 3, 20), colors="blue")
plt.savefig("contorno.pdf")

plt.show()



