import tensorflow as tf

#Ejecutar el script desde Jupyter o Anaconda Prompt. Desde cmd de Windows da error.
#pip install tensorflow==2.0.0.  
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
