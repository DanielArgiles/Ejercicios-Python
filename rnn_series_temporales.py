# Ejemplo de red neuronal recurrente (RNN) mediante series temporales
# Código desarrollado con versión de TensorFlow anterior a 2.0

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

leche= pd.read_csv('produccion_leche.csv',index_col='Month')
print(leche.head()) # Leemos las 5 primeras filas
print(leche.info())
leche.index=pd.to_datetime(leche.index)
leche.plot()
plt.show()

# Dividimos las 168 entradas del dataframe en 150 para entrenamiento y 18 para pruebas o test
conjunto_entrenamiento= leche.head(150)
conjunto_pruebas=leche.tail(18)
print(conjunto_entrenamiento)
print(conjunto_pruebas)

# Normalizamos los datos para poder usar tensorflow
from sklearn.preprocessing import MinMaxScaler
normalizacion=MinMaxScaler()
entrenamiento_normalizado=normalizacion.fit_transform(conjunto_entrenamiento)
pruebas_normalizado=normalizacion.transform(conjunto_pruebas)

def lotes(datos_entrenamiento,tamaño_lote,pasos):
    comienzo=np.random.randint(0,len(datos_entrenamiento)- pasos)
    lote_y=np.array(datos_entrenamiento[comienzo:comienzo+pasos+1]).reshape(1,pasos+1)
    return lote_y[:,:-1].reshape(-1,pasos,1),lote_y[:,1:].reshape(-1,pasos,1)
    
numero_entradas=1
numero_pasos=18
numero_neuronas=120
numero_salidas=1
tasa_aprendizaje=0.001
numero_iteraciones=5000
tamaño_lote=1

x=tf.compat.v1.placeholder(tf.float32,[None,numero_pasos,numero_entradas])
y=tf.compat.v1.placeholder(tf.float32,[None,numero_pasos,numero_salidas])

# tf.contrib no funciona en TensorFlow 2.0 
capa=tf.contrib.rnn.OutputProjctionWrapper(tf.contrib.rnn.BasicLSTMcell(num_units=numero_neuronas,activation=tf.nn.relu),output_size=numero_salidas)
salidas,estados=tf.nn.dynamic_rnn(capa,x,dtype=tf.float32)
funcion_error=tf.reduce_mean(tf.square(salidas-y))
optimizador=tf-train.AdamOptimizer(learning_rate=tasa_aprendizaje)
entrenamiento=optimizador.minimize(funcion_error)
init=tf.compat.v1.global_variables_initializer()
saver=tf.train.Saver()

with tf.compat.v1.Session() as sesion:
    sesion.run(init) # Inicializamos las variables
    for iteracion in range (numero_iteraciones_entrenamiento):
        lote_x,lote_y= lotes[entrenamiento_normalizado,tamaño_lote,numero_pasos]
        sesion.run(entrenamiento,feed_dict={x:lote_x,ylote_y})
        if iteracion %100==0
            error=funcion_error.eval(feed_dict={x:lote_x,ylote_y})
            print(iteracion,"\t Error",error)
        saver.save(sesion,"./modelo_series_temporales")
        
# Predición para los datos de test
with tf.compat.v1.Session() as sesion:
    saver.restore(sesion,"./modelo_series_temporales")
    entrenamiento_seed=list(entrenamiento_normalizado[-18:])
    for iteracion in range(18):
        lote_x=np.array(entrenamiento_seed[-numero_pasos:]).reshape(1,numero_pasos,1)
        prediccion_y=sesion.run(salidas,feed_dict={x:lote_x})
        entrenamiento_seed.append(prediccion_y[0,-1,0])
  
resultados=normalizacion.inverse_transform(np.array(entrenamiento_seed[18:]).reshape(18,1)) #18 filas,1 columna
conjunto_pruebas['Predicciones']= resultados

# Relación gráfica
conjunto_pruebas.plot()
plt.show()
