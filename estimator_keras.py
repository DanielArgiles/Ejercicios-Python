# Estimator, API de tensorflow que simplifica la programación en temas de Machine Learning

import tensorflow as tf
from tensorflow import estimator
from sklearn.datasets import load_wine

vino= load_wine()
print(vino['DESCR']) #Información del dataset
caracteristicas=vino['data']
objetivo=vino['target']

# Dividimos en datos de entrenamiento y en datos de pruebas (test)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(caracteristicas,objetivo,test_size=0.3) # Un 30% de los datos serán de prueba y un 70% de entrenaiento

# Normalizamos los datos para poder trabajar con tensorflow
from sklearn.preprocessing import MinMaxScaler
normalizador=MinMaxScaler()
x_train_normalizado=normalizador.fit_transform(x_train)
x_test_normalizado=normalizador.transform(x_test)
print(x_train_normalizado.shape) # filas 124 y columnas 13

columnas_caracteristicas=[tf.feature_column.numeric_column('x',shape=[13])] # 13 columnas
# Hasta aquí será igual para Estimator API y Keras

# Creamos el modelo de entrenamiento (de forma diferente para Estimator API y Keras)
modelo=tf.compat.v1.estimator.DNNClassifier(hidden_units=[20,20,20],feature_columns=columnas_caracteristicas,n_classes=3,optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01))

# Creamos la función de entrada
funcion_entrada=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x':x_train_normalizado},y=y_train,shuffle=True,batch_size=10,num_epochs=10)

# Entrenamos el modelo
modelo.train(input_fn=funcion_entrada,steps=600)

# Predicciones
funcion_evaluacion=tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x':x_test_normalizado},shuffle=False) # no le pasamos y porque es lo que queremos predecir
predicciones=list(modelo.predict(input_fn=funcion_evaluacion))
predicciones_finales= [  p['class_ids'][0] for p in predicciones]

# Evaluamos, y comparamos valores reales de test con las predicciones
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predicciones_finales))

#---#
"""
Keras, biblioteca de redes neuronales, de código abierto escrito en Python
Es capaz de ejecutar en TensorFlow, en Microsoft cognitive o Theano
Diseñada para experimentar de forma rápida con redes Deep Learning
"""

"""Con TensorFlow 1.0)
from tensorflow.contrib.keras import models,layers,losses,optimizers,metrics,activations
modelo=models.Sequential()
modelo.add(layers.Dense(units=13,input_dim=13,activation='relu'))
modelo.add(layers.dense(units=13,activation='relu'))
modelo.add(layers.dense(units=13,activation='relu'))
modelo.add(layers.Dense(units=3),activation='softmax')
"""

# Con Tensorflow 2.0
# Construimos un modelo tf.keras.Sequential apilando capas.
# Los bloques de construccion básicos de una red neuronal son las capas o layers.
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(13, input_dim=13, activation='relu'), # 13 nodos o neuronas
tf.keras.layers.Dense(13,  activation='relu'), # 13 nodos o neuronas
tf.keras.layers.Dense(13,  activation='relu'), # 13 nodos o neuronas
tf.keras.layers.Dense(3,  activation='softmax'), # 3 nodos o neuronas 
])

# Compilamos el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos el modelo
model.fit(x_train_normalizado, y_train, epochs=60)

# Predicciones
predicciones = model.predict_classes(x_test_normalizado) # No pasamos y porque queremos estimarla

# Evaluamos el modelo para ver nuestra precisión
from sklearn.metrics import classification_report
print(classification_report(y_test,predicciones))
