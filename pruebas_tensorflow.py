import tensorflow as tf

#Ejecutar el script desde Jupyter o Anaconda Prompt. Desde cmd de Windows da error.
#pip install tensorflow==2.0.0. Vemos versión con : print(tf.__version__) 
#conda create –n pruebastensorflow Python=3.5 tensorflow=1 jupyter numpy pandas scikit-learn matplotlib.
#conda activate pruebastensorflow.

mensaje1=tf.constant("Hola")
mensaje2=tf.constant("mundo")
a= tf.constant(10)
b= tf.constant(5)
constante=tf.constant(15)
matriz1=tf.fill((6,6),10)
matriz2= tf.random.normal((5,5))
matriz3= tf.random.uniform((4,4), minval=0, maxval=5)
matriz_ceros=tf.zeros((2,2))
matriz_unos=tf.ones((3,3))
operaciones=[constante,matriz1,matriz2,matriz3,matriz_ceros,matriz_unos]


# Grafos en TensorFlow (nodo 3= nodo 1 (operación) nodo 2)
with tf.compat.v1.Session() as sesion:
    resultado1=sesion.run(mensaje1 + mensaje2)
    print(resultado1)
    
with tf.compat.v1.Session() as sesion:
    resultado2=sesion.run(a+b)
    print(resultado2)
    
with tf.compat.v1.Session() as sesion:
    for op in operaciones:
        print(sesion.run(op))
        print("\n")
        
# Crear un grafo por defecto  
 print(tf.compat.v1.get_default_graph())

# Crear nuestro propio grafo
grafo1=tf.Graph()
  
# Conseguir que nuestro grafo sea grafo por defecto para una serie de operaciones
with grafo1.as_default():
    print(grafo1 is tf.compat.v1.get_default_graph()) # Imprime True
print(grafo1 is tf.compat.v1.get_default_graph()) # Imprime False . Nuestro propio grafo no será grafo por defecto

# Variables
tensor= tf.random.uniform((5,5),0,1) #  tensor aleatorio con 5 filas y 5 columnas, valor mínimo el 0 y valor máixmo el 1
variable=tf.Variable(initial_value=tensor) # Asignamos una variable que es el valor de tensor
print(variable)

inicializador=tf.compat.v1.global_variables_initializer() # Las variables necesitan ser siempre inicializadas antes de ser utilizadas en una sesión de tensorflow.
with tf.compat.v1.Session() as sesion:
    sesion.run(inicializador) # Inicializamos las variables
    resultado=sesion.run(variable) # Ejecutamos la variable
print(resultado) # Mostramos la variable

# Placeholders están inicialmente vacíos y se utilizan para alimentar los ejemplos de entrenamiento del modelo.
# Es una especie de incógnita dentro de las ecuaciones.
# Los placeholders no hace falta inicializarlos para usarlos.
incognitas= tf.compat.v1.placeholder(tf.float32, shape= (20,20))
print(incognitas)

