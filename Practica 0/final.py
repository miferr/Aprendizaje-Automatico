
#   Librerias a usar
import scipy
import time
import random as r
import numpy as np
import matplotlib as plt
from pylab import *
from scipy import integrate

#   Funcion de prueba
def cuadrado(x):
    return x * x

#   Funcionamiento: I = (Ndebajo / Ntotal) * (b - a) * M

#   Sin uso de vectores
def integra_mc(fun, a, b, num_puntos=1000000):
    M = fun(b)
    base = b - a
    nDebajo = 0
    for i in range(num_puntos):
        x = r.uniform(a,b)
        y = r.uniform(0,M)
        if(y <= fun(x)): 
            nDebajo += 1
    return (nDebajo/num_puntos) * base * M

#   Usando vectores
def numpy_integra_mc(fun, a, b, num_puntos=1000000):
    M = fun(b)
    base = b - a
    x = np.random.uniform(a,b, num_puntos)
    y = np.random.uniform(0,M, num_puntos)
    nDebajo = sum(y <= fun(x))
    return (sum(y <= fun(x))/num_puntos) * base * M

#   Funcion para representar el metodo
def ejemplo(fun, a, b, num_puntos=100000):  
    M = fun(b)
    for i in range(num_puntos):
       x = r.uniform(a,b)
       y = r.uniform(0,M)

       #Solo se printean algunos puntos
       if (0 == r.randint(0,100)):
           plt.plot(x,y, marker="x", color="red")
        
    plt.xlim(0, 6)
    plt.ylim(0, 27)
    x = arange(0, 7, 1)
    plt.plot(x**2)
    plt.legend(["""    La grafica 
    se ha hecho
    con 1 de cada 
    cien puntos 
    creados para que 
    sea visible"""])

#   Calculamos el tiempo de ejecucion de cada funcion y su resultado
ejemplo(cuadrado, 0, 5)
init = time.time()
s = integra_mc(cuadrado, 0, 5)
s_time = time.time() - init

init = time.time()
c = numpy_integra_mc(cuadrado, 0, 5)
c_time = time.time() - init

print("Resultado sin vectores("+str("{0:.2f}".format(s_time))+"s): "+ str("{0:.2f}".format(s)))
print("Resultado con vectores("+str("{0:.2f}".format(c_time))+"s): "+ str("{0:.2f}".format(c)))
print("Resultado con scypy: "+ str("{0:.2f}".format(scipy.integrate.quad(cuadrado, 0, 5)[0])))
print("La funcion con bucle tarda " + str("{0:.2f}".format(s_time/c_time)) + " veces más que con vectores")
#n_puntos = 10000
#print("Nuestro resultado: " + str(integra_mc(cuadrado, 0, 5, n_puntos)) + " y el resultado de sicpy: " + str(integrate.quad(cuadrado, 0, 5)[0]))