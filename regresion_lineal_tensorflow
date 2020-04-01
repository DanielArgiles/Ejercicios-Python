import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Ejemplo de regresión lineal (y= mx + b) con TensorFlow.
datos_x= np.linspace(0,10,10)+ np.random.uniform(-1,1,10) # sumamos un pequeño ruido
datos_y= np.linspace(0,10,10)+ np.random.uniform(-1,1,10) # sumamos un pequeño ruido
plt.plot(datos_x,datos_y,'*')
plt.show()

print(np.random.rand(2)) #Generamos dos números aleatorios y obtenemos 0.99 y 0.098
m=tf.Variable(0.99)
b=tf.Variable(0.098)

error=0
for x,y in zip(datos_x,datos_y):
    y_pred=m*x+b #valor de predicción de y (y es valor real)
    error=error +(y-y_pred)**2 
    
# Ahora disminuiremos al máximo el error con un optimizador
optimizador= tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
entrenamiento= optimizador.minimize(error)

inicializacion=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sesion:
    sesion.run(inicializacion)
    pasos= 1 
    for i in range(pasos): # bucle que se ejecuta una vez solo porque  pasos es 1
        sesion.run(entrenamiento) # ejecutamos el entrenamiento que optimiza el error
    final_m,final_b=sesion.run([m,b]) # Nuevos valores de m y b obtenidos con el optimizador
    
x_test=np.linspace(-1,11,10) # Nueva x con 10 valores entre -1 y 11
y_pred_2= (final_m * x_test) + final_b # Nueva y de predicción a partir de nuevos valores m,x,b

# Representamos una linea de regresión simple y lineal roja optimizada para reducir el error a partir de los puntos azules iniciales.
plt.plot(x_test,y_pred_2,'r')
plt.plot(datos_x,datos_y,'*') 
plt.show()
