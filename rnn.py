# Ejemplo de red neuronal recurrente (RNN) con una capa de 3 neuronas desenrollada 2 veces. 
# Capa 1 (placeholder entrada x0, peso entrada Wx, salida y0= Wy), Capa 2 (placeholder entrada x1, pesos entrada Wx,Wy, salida y1).

import numpy as np
import tensorflow as tf

numero_entradas= 2 # 2 entradas= 2 capas
numero_neuronas= 3

tf.compat.v1.disable_eager_execution()
x0= tf.compat.v1.placeholder(tf.float32,[None, numero_entradas]) # no conocemos el nímero de filas, el numero de columnas será el número de entradas
x1= tf.compat.v1.placeholder(tf.float32,[None, numero_entradas])

Wx= tf.Variable(tf.compat.v1.random_normal(shape=[numero_entradas,numero_neuronas]))
Wy= tf.Variable(tf.compat.v1.random_normal(shape=[numero_neuronas,numero_neuronas]))
b= tf.Variable(tf.zeros([1,numero_neuronas])) # variable bias,  1 fila, columnas=numero_neuronas

y0= tf.tanh(tf.matmul(x0,Wx)+b)
y1= tf.tanh(tf.matmul(y0,Wy)+tf.matmul(x1,Wx)+b)

lote_x0= np.array([[0,1],[2,3],[4,5]])
lote_x1= np.array([[2,4],[3,9],[4,1]])

init=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sesion:
    sesion.run(init) # Inicializamos las variables
    salida_y0,salida_y1= sesion.run([y0,y1],feed_dict={x0:lote_x0,x1:lote_x1}) # feed_dict para valores de los placeholders
    
print(salida_y0)
print('\n')
print(salida_y1) 
