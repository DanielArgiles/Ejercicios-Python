"""
Clasificador de imágenes con el conjunto de datos (dataset) CIFAR-10
------
El problema de la identificación automática de objetos en las fotografías es dificil debido al número casi infinito de permutaciones de objetos, posiciones, iluminación, etc. 
Este es un problema bien estudiado en la visión por computador y, más recientemente, una importante demostración de la capacidad del Deep Learning o aprendizaje profundo. 
El Instituto Canadiense de Investigaciones Avanzadas elaboró un conjunto de datos estándar sobre visión por computador y aprendizaje profundo para este problema (CIFAR-10).
Consta de 60.000 fotos divididas en 10 clases (de ahí el nombre CIFAR-10). 
El conjunto de datos se divide de forma estándar, donde se utilizan 50.000 imágenes para la formación de un modelo y los 10.000 restantes para evaluar su desempeño. 
Las imágenes están en los 3 canales (rojo, verde y azul) y son cuadrados pequeños que miden 32 x 32 píxeles.
Las 10 clases diferentes de este conjunto de datos son:
Avión, Coche, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión.
El conjunto de datos CIFAR-10 ya está disponible en el módulo de conjuntos de datos de Keras. 
No necesitamos descargarlo,sino que podemos importarlo directamente desde keras.datasets.
------
La idea del proyecto es construir un modelo de clasificación de imágenes que pueda identificar a que clase pertenece una imagen de entrada. 
La clasificación de imágenes se usa en muchas aplicaciones y es un gran proyecto para comenzar el Deep Learning (aprendizaje profundo).
Específicamente, la clasificación de imágenes se incluye en la categoría de proyectos de visión por computadora.
Primero, exploraremos nuestro conjunto de datos, y luego entrenaremos nuestra red neuronal convolucional usando Keras.
La clasificación de imágenes es una aplicación tanto de clasificación supervisada como de clasificación no supervisada.
Construiremos una interfaz gráfica GUI utilizando la biblioteca Tkinter de python para cargar las imágenes que queramos.
------
Podemos ejecutar todo el código en jupyter notebook (separando en 2 partes), ó guardar en 2 scripts diferentes (.py), el modelo de entrenamiento por un lado, y la interfaz gráfica por otro y ejecutarlos por separado.
De esta segunda forma, ahorraremos el tiempo de tener que entrenar el modelo cada vez que queramos usar la interfaz gráfica, ya que una vez se entrena el modelo se guarda en un archivo "model1_cifar_10epoch.h5". 
"""
# Simple modelo CNN para el conjunto de datos de CIFAR-10 

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras import backend as K
#K.set_image_dim_ordering('th')


# semilla del número aleatorio para asegurarnos de que los resultados son reproducibles.
seed = 7
numpy.random.seed(seed)

# cargamos el conjunto de datos CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# Mostramos una malla de fotografías de 3 x 3 del conjunto de datos. 
# Las imágenes se han ampliado a partir de su pequeño tamaño de 32 x 32, pero se pueden ver claramente algunos animales y automóviles. 
# También puede ver cierta distorsión en las imágenes que se han forzado a la relación de aspecto cuadrado

n=9
# plt.figure(figsize=(20,10)), ampliamos la imagen
for i in range(n):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i*8]) # coge imágenes de 8 en 8 según el conjunto de imágenes
# show the plot
pyplot.show()  


# Exploramos el set de datos
print(X_train.shape)
print(len(y_train))
print(y_train)
print(X_test.shape)
print(len(y_test))


# Los valores de los píxeles están en el rango de 0 a 255 para cada uno de los canales rojo, verde y azul. 
# Es una buena práctica trabajar con datos normalizados por eso dividimos entre 255.0.
# Los datos se cargan como enteros, por lo que debemos lanzarlos a valores de coma flotante para poder realizar la división.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Las variables de salida se denotan como un vector de números enteros de 0 a 1 para cada clase.
# Podemos usar una codificación en caliente para transformarlos en una matriz binaria con el fin de modelar mejor el problema de la clasificación. 
# Sabemos que hay 10 clases para este problema, así que podemos esperar que la matriz binaria tenga un ancho de 10.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Creacion del modelo

# Comencemos definiendo una estructura simple de CNN como línea de base y evaluemos como funciona.
# 1- Capa de entrada convolucional, 32 mapas de características con un tamaño de 3 x 3, una activación de rectier y una restricción de peso de la norma máxima establecida en 3.
# 2- Abandono fijado en el 20%.
# 3- Capa convolucional, 32 mapas de características con un tamaño de 3 x 3, una función de activación de rectier y una restricción de peso de la norma máxima establecida en 3.
# 4- Capa máxima de pool con el tamaño 2 x 2.
# 5- Capa Flatten.
# 6- Capa totalmente conectada con 512 unidades y función de activación de rectier.21.3. CNN simple para CIFAR-10 152.
# 7- Abandono al 50%.
# 8- Capa de salida totalmente conectada con 10 unidades y función de activación softmax.
# Se utiliza una función de pérdida logarítmica con el algoritmo de optimización de descenso de gradiente estocástico configurado con un gran impulso y disminución de peso, comenzando con una tasa de aprendizaje de 0,01. 
# A continuación se proporciona una visualización de la estructura de la red.

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu',
kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compilamos el modelo
epochs = 10
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# Ajustamos este modelo con 25 épocas (epochs) y un tamaño de lote de 32 (batch_size). 
# Se eligió un pequeño número de épocas de forma rápida. 
# Normalmente, el número de épocas sería uno o dos órdenes de magnitud mayor para este problema
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Final evaluacion del modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("Precision: %.2f%%" % (scores[1]*100))


"""
Consejos para mejorar el rendimiento del modelo:


-Entrena para más épocas: Cada modelo fue entrenado para un número muy pequeño de épocas, 25. Es común entrenar grandes redes neuronales convolucionales para cientos o miles de épocas. Yo esperaría que las mejoras en el rendimiento puedan lograrse aumentando de manera significativa el número de épocas de entrenamiento.
-Aumento de datos de imagen: Los objetos de la imagen varían en su posición. Otra mejora es probable que se pueda lograr con un aumento en el rendimiento del modelo mediante el uso de algún tipo de aumento de datos. Métodos como la estandarización y los desplazamientos aleatorios de la imagen pueden ser beneficiosos.
-Topología de red más profunda: La red más grande que se presenta es profunda, pero incluso redes mas profundas se podrían diseñar para el problema. Esto puede implicar más mapas de características y quizás un pooling menos agresivo. Además, las topologías de red convolucional estándar que han demostrado ser útiles pueden ser adoptadas y evaluadas sobre este problema.
 
"""
# Save the model
model.save("model1_cifar_10epoch.h5")




# GUI - Interfaz Gráfica

#sudo apt-get install python3-tk

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy

# Cargamos el modelo de aprendizaje para clasificar las imágenes

from keras.models import load_model
model = load_model('model1_cifar_10epoch.h5')

# Diccionario para etiquetar todas las clases del conjunto de datos CIFAR-10.
classes = { 
    0:'Avión',
    1:'Coche',
    2:'Pájaro',
    3:'Gato',
    4:'Ciervo',
    5:'Perro',
    6:'Rana',
    7:'Caballo',
    8:'Barco',
    9:'Camión' 
}


# Inicializamos la GUI

top=tk.Tk()
top.geometry('800x600')
top.title('Image Classification CIFAR10')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
    
def show_classify_button(file_path):
    classify_b=Button(top,text="Clasificar Imágen",
   command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
    
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Cargar una imágen",command=upload_image,
  padx=10,pady=5)
upload.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Clasificador de Imágenes con CIFAR10",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()

top.mainloop()




