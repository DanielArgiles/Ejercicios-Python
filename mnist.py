import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Cargamos y preparamos el conjunto de datos MNIST. Convertimos los ejemplos de números enteros a números de punto flotante.
mnist = tf.keras.datasets.mnist  # Ejemplo de otro set de imágenes de MNIST: fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # Ejemplo de otro set de imágenes de MNIST: fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

#Exploramos el set de datos
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))

# Pre-procesamos el set de datos. Los valores de los pixeles de la primera imagen del set , están entre 0 y 255.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Cada imagen es mapeada a una única etiqueta. class_names no está incluido en el dataset mnist, así que lo creamos.
class_names = ['cero', 'uno', 'dos', 'tres', 'cuatro',
               'cinco', 'seis', 'siete', 'ocho', 'nueve']
               
# Verificamos que el set de datos está en el formato adecuado, mostrando las 25 primeras imagánes y su nombre de clase.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Construimos un modelo tf.keras.Sequential apilando capas.
# Los bloques de construccion básicos de una red neuronal son las capas o layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # transforma el formato de las imágenes de un array bi-dimensional (de 28 por 28 pixeles) a un array uni dimensional (de 28*28 pixeles = 784 pixeles)
  # La capa tf.keras.layers.Flatten no tiene parámetros que aprender, solo reformatea el set de datos.
  tf.keras.layers.Dense(128, activation='relu'), # 128 nodos o neuronas
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax') # 10 nodos o neuronas
])

# Compilamos el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
# Entrenamos y evaluamos el modelo
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Hacemos predicciones
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

# Funciones plot_image y plot_value_array
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
# Mostramos una imagen individual , seleccionando con el valor de i. 
# Las etiquetas de predicción correctas están en azul y las incorrectas en rojo. 
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Mostramos múltiples imágenes con sus predicciones.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#  Usamos el modelo entrenado para hacer una prediccion sobre una única imagen.
img = test_images[1]
print(img.shape)
# Los modelos de tf.keras son optimizados sobre batch o bloques.
img = (np.expand_dims(img,0))
print(img.shape)

# Ahora predecimos la etiqueta correcta para esta imagen
predictions_single = model.predict(img) # model.predict retorna una lista de listas para cada imagen dentro del batch o bloque de datos.
print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])


# Para tensorflow 1.0: 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MINST',one_hot=True) # Importamos los 4 ficheros que incluyen las imagenes de entrenamiento y de test y las etiquetas
imagen=mnist.train.images[1]
mnist.train.num_examples
mnist.test.num_examples
type(mnist)
imagen=imagen.reshape(28,28) # matriz 28 x 28
plt.imshow(imagen)
