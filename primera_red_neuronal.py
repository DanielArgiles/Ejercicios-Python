import tensorflow as tf
import numpy as np


# Ejemplo para recordar el uso de las variables y placeholders

aleatorio_a=np.random.uniform(0,50,(4,4))
print(aleatorio_a)
aleatorio_b=np.random.uniform(0,50,(4,1))
print(aleatorio_b)
tf.compat.v1.disable_eager_execution()
a=tf.compat.v1.placeholder(tf.float32)
b=tf.compat.v1.placeholder(tf.float32)
suma=a+b
multiplicacion=a*b

with tf.compat.v1.Session() as sesion:
    resultado_suma=sesion.run(suma,feed_dict={a:10,b:20}) # con el diccionario damos valores
    print(resultado_suma)
with tf.compat.v1.Session() as sesion:
    resultado_suma=sesion.run(suma,feed_dict={a:aleatorio_a,b:aleatorio_b}) # con el diccionario damos valores
    print(resultado_suma)
with tf.compat.v1.Session() as sesion:
    resultado_multiplicacion=sesion.run(multiplicacion,feed_dict={a:10,b:20}) # con el diccionario damos valores
    print(resultado_multiplicacion)
with tf.compat.v1.Session() as sesion:
    resultado_multiplicacion=sesion.run(multiplicacion,feed_dict={a:aleatorio_a,b:aleatorio_b}) # con el diccionario damos valores
    print(resultado_multiplicacion)

# Mi primera red neuronal: Z(función activación)= W(variable)*X(Placeholder)+B(variable)

caracteristicas=10
neuronas=4
x=tf.compat.v1.placeholder(tf.float32,(None,caracteristicas)) # None: desconocemos Número de filas o datos en juego de pruebas
# si conocemos las columnas, que será las características
w=tf.Variable(tf.random.normal([caracteristicas,neuronas])) #filas,columnas
b= tf.Variable(tf.ones([neuronas]))
multiplicacion=tf.matmul(x,w) # matmul es multiplicación de matrices
z=tf.add(multiplicacion,b) # add es suma de matrices
activacion=tf.sigmoid(z)

inicializador=tf.compat.v1.global_variables_initializer() # Para ejecutar la función de activación, tenemos que inicializar las variables
valores_x=np.random.random([1,caracteristicas])
print(valores_x)

with tf.compat.v1.Session() as sesion:
    sesion.run(inicializador) # Inicializamos las variables
    resultado=sesion.run(activacion,feed_dict={x:valores_x}) # Ejecutamos la variable
print(resultado) # Mostramos la variable , valores entre 0 y 1
