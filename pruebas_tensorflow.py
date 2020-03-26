import tensorflow as tf

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



with tf.compat.v1.Session() as sesion:
    resultado1=sesion.run(mensaje1 + mensaje2))
    print(resultado1)
    
with tf.compat.v1.Session() as sesion:
    resultado2=sesion.run(a+b)
    print(resultado2)
    
with tf.compat.v1.Session() as sesion:
    for op in operaciones:
        print(sesion.run(op))
        print("\n")
