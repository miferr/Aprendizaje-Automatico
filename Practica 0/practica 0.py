import numpy as np
import matplotlib as plt
from pylab import *
import random
from scipy import integrate

def cuadrado(x):
    return x*x

def integra_mc(fun, a, b, num_puntos=10000):
    figure(figsize=(15,6))
    
    M = fun(b)
    base = b - a
    nDebajo = 0
    for i in range(num_puntos):
        x = random.uniform(a, b)
        y = random.uniform(0,M)
        #Solo se printean algunos puntos
        if (0 == random.randint(0,10)):
            plt.plot(x,y, marker="x", color="red")
            
        if(y <= fun(x)): 
            nDebajo += 1
            
    #Parte para graficar
    plt.xlim(0, 6)
    plt.ylim(0, 27)
    x = arange(0, 7, 1)
    plt.plot(x**2)
    plt.legend(["""    La grafica 
    se ha hecho
    con 1 de cada 
    diez puntos 
    creados para que 
    sea visible"""])
    #Fin de la grafica
        
    return (nDebajo/num_puntos) * base * M

resultado = "Nuestro resultado: " + str(integra_mc(cuadrado, 0, 5)) + " y el resultado de sicpy: " + str(integrate.quad(cuadrado, 0, 5)[0])
print(resultado)

# Parte de Toni
#   Practica 0 

#   Librerias a usar
import random
import scipy
import numpy
import time

#   Funcion de prueba
def cuadrado(x):
    return x * x

#   Funcionamiento: I = (Ndebajo / Ntotal) * (b - a) * M

#   Sin uso de vectores
def integra_mc(fun, a, b, num_puntos=100000):
    M = fun(b)
    base = b - a
    nDebajo = 0
    for i in range(num_puntos):
        x = random.uniform(a,b)
        y = random.uniform(0,M)
        if(y <= fun(x)): 
            nDebajo += 1
    return (nDebajo/num_puntos) * base * M

#   Usando vectores
def numpy_integra_mc(fun, a, b, num_puntos=100000):
    M = fun(b)
    base = b - a
    nDebajo = 0
    x = numpy.random.uniform(a,b, num_puntos)
    y = numpy.random.uniform(0,M, num_puntos)
    nDebajo = sum(y <= fun(x))
    return (nDebajo/num_puntos) * base * M

#   Calculamos el tiempo de ejecucion de cada funcion y su resultado
init = time.time()
s = integra_mc(cuadrado, 0, 5)
s_time = time.time() - init

init = time.time()
c = numpy_integra_mc(cuadrado, 0, 5)
c_time = time.time() - init

print("Resultado sin vectores("+str(s_time)+"s): "+ str(s))
print("Resultado con vectores("+str(c_time)+"s): "+ str(c))
print("Resultado con scypy: "+ str(scipy.integrate.quad(cuadrado, 0, 5)[0]))
