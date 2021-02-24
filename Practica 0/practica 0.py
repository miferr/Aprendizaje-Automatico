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