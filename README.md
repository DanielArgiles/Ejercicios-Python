# Ejercicios-Python

## Ejercicio 1 : Catálogo de películas persistente
En este ejercicio se crea un catálogo de películas persistente (catalogo_peliculas_persistente.py).
Mediante la programación orientada a objetos se añaden y eliminan diferentes películas, con la peculiaridad de que al hacer uso del módulo pickle,  todos los cambios realizados en dicho catálogo se guardan en un fichero binario (catalogo.pckl), de tal manera que cuando se cierre el programa, los datos persistirán.

## Ejercicio 2 : Editor de texto con tkinter
En este ejercicio se crea un pequeño editor de texto (editor_texto.py) utilizando para la interfaz gráfica el módulo tkinter.
Widgets: Tk(raíz), Menu, Text.
A través de un menú desplegable se pueden abrir, guardar y crear nuevos ficheros de texto.

## Ejercicio 3 : Gestor de platos del menú de un restaurante con SQLite
En este ejercicio se crea un gestor de categorías y platos del menú de un restaurante (restaurante_rincondelvino.py) mediante una base de datos (restaurante_rincondelvino.db) creada con SQLite.
Inspector externo: DB Browser for SQLite.

## Ejercicio 4 : Menú de un restaurante con tkinter y SQLite
En este ejercicio se crea la interfaz gráfica del menú de un restaurante (menu_restaurante_rincondelvino.py).
El programa se conecta a la base de datos del Ejercicio 3 (restaurante_rincondelvino.db) para buscar y añadir la lista de categorías y platos.

## Ejercicio 5 : Lectura de Hoja de Cálculo Excel (xlsx) con Pandas (Data Science)
En este ejercicio se realiza algunas pruebas de selección e impresión de las diferentes filas y columnas de un Libro de Excel (clientes.xlsx), haciendo uso de la librería Pandas.
Inicialmente se transforma cada Hoja del Libro de Excel en un DataFrame, para después leer los datos con los métodos iloc y loc.
Script: clientes_pandas.py.

## Ejercicio 6 : Lectura de un fichero de Excel (csv) con Pandas (Data Science)
En este ejercicio se realiza algunas pruebas de tratamiento de datos de un fichero csv (valores separados por comas) con el nombre personas.csv, mediante la librería Pandas.
Script: personas_pandas.py

## Ejercicio 7 : Pruebas con NumPy (Data Science)
En este ejercicio se realiza algunas pruebas básicas con NumPy.
Script: pruebas_numpy.py

## Ejercicio 8 : Pruebas con Matplotlib (Visualización de datos)
En este ejercicio se realiza algunas pruebas de visualización de datos.
Script: pruebas_matplotlib.py

## Ejercicio 9 : Pruebas con Scikit Learn (Machine Learning)
En este ejercicio se realiza unas pruebas para dividir un conjunto de datos disponible en datos para entrenamiento y en datos para test.
Script: pruebas_scikitlearn.py

## Ejercicio 10 : Detector de colores de una imagen con Pandas (Data Science) y OpenCV (Visión Artificial)
En este ejercicio se construye una aplicación a través de la cual se puede obtener automáticamente el nombre del color haciendo doble clic en las diferentes zonas de una imagen (color_animales.jpg).
Nombre del script: detector_colores.py.
Tenemos un archivo de datos (colores.csv) que incluye 865 nombres de colores junto con sus valores RGB y hexadecimales, sabiendo que los colores se componen de 3 colores primarios; rojo, verde y azul (RGB). 
Para obtener el nombre del color correctamente, calculamos una distancia (d) que nos dice como de cerca estamos del color y elegimos el que tiene la distancia mínima.
d = abs (Rojo - ithRedColor) + (Verde - ithGreenColor) + (Azul - ithBlueColor).

## Ejercicio 11 : Pruebas con TensorFlow (Deep Learning)
En este ejercicio se realiza algunas pruebas de sintáxis básica de TensorFlow.
Script: pruebas_tensorflow.py

## Ejercicio 12 : Mi primera red neuronal con TensorFlow y NumPy
En este ejercicio se realiza un ejemplo de red neuronal: Z=W*X+B. 
Script: primera_red_neuronal.py

## Ejercicio 13 : Regresión lineal simple con TensorFlow, NumPy y Matplotlib
En este ejercicio se realiza un ejemplo de regresión lineal simple: y=mx+b.
Script: regresion_lineal_simple_tf.py

## Ejercicio 14 : Bibliotecas: Estimator API y Keras (Deep Learning)
Ejemplos básicos sobre como utilizar Estimator API y Keras. 
Script: estimator_keras.py

## Ejercicio 15 : Ejemplo de algoritmo de clasificación con TensorFlow, Pandas y Scikit Learn
En este ejercicio se realiza un ejemplo de un algoritmo de clasificación (aprendizaje supervisado).
A partir de los datos de un fichero de Excel (csv) llamado ingresos.csv, se predicen los ingresos de una persona ((<=50 K) = 0, (>50 K)=1) en función de sus características(edad, género, etc).
Script: clasificacion_tf.py

## Ejercicio 16 : Ejemplo de algoritmo de regresión con TensorFlow, Pandas y Scikit Learn
En este ejercicio se realiza un ejemplo de un algoritmo de regresión (aprendizaje supervisado).
A partir de los datos de un fichero de Excel (csv) llamado precios_casas.csv, se predice el valor medio de una casa en función de sus características(habitaciones, latitud, longitud, etc).
Script: regresion_tf.py

## Ejercicio 17 : Ejemplo de Red Neuronal Convolucional (CNN) con MNIST, TensorFlow, Keras, NumPy y Matplotlib
En este ejercicio se realiza un ejemplo de red neuronal convolucional (CNN), entrenando un modelo secuencial para clasificar y predecir dígitos escritos a mano. Para ello utilizamos MNIST (base de datos de imágenes de dígitos escritos a mano).
Script: cnn_mnist.py

## Ejercicio 18 : Ejemplo de Red Neuronal Recurrente (RNN) con TensorFlow y NumPy
En este ejercicio se realiza un ejemplo de red neuronal recurrente (RNN) con una capa de 3 neuronas desenrollada 2 veces.
Capa 1 (placeholder entrada x0, peso entrada Wx, salida y0= Wy), Capa 2 (placeholder entrada x1, pesos entrada Wx,Wy, salida y1).
Script: rnn.py

## Ejercicio 19 : Ejemplo de Red Neuronal Recurrente (RNN) mediante series temporales con TensorFlow, NumPy, Pandas y Matplotlib
Pendiente de revisar y migrar el código TensorFlow 1.X a TensorFlow 2.0.
Script: rnn_series_temporales.py, Archivo Excel: produccion_leche.csv

## Ejercicio 20 : Test Driven development(TDD): Doctest
Ejercicios con el módulo doctest.
Script: doctest.py

## Ejercicio 21 : Unittest
Ejercicios de unit testing con el módulo unittest.
Script: unittest.py

## Ejercicio 22 : Pruebas de dataframes en Apache Spark con PySpark
En este ejercicio se realiza pruebas de conceptos básicos de dataframes en Apache Spark, así como operaciones básicas, operaciones Aggregate y Groupby, missing data, dates and timestamps.
Archivos necesarios: appl_stock.csv, ContainsNull.csv, sales_info.csv,people.json, walmart_stock.csv,
Script: pruebas_dataframes_spark.py

## Ejercicio 23 : Ejemplos de Regresión Lineal en Apache Spark con PySpark y MLlib (Spark para Machine Learning)
En este ejercicio se realiza algunos ejemplos de regresión lineal así como un proyecto real de consultoría en Apache Spark con PySpark y MLlib.
Archivos necesarios: sample_linear_regression_data.txt, Ecommerce_Customers.csv, cruise_ship_info.csv, 
Script: regresion_lineal_spark.py

## Ejercicio 24 : Ejemplos de Regresión Logística (clasificación) en Apache Spark con PySpark y MLlib (Spark para Machine Learning)
En este ejercicio se realiza algunos ejemplos de regresión logística (clasificación) así como un proyecto real de consultoría en Apache Spark con PySpark y MLlib.
En el segundo ejemplo se aprende a desarrollar el código en Databricks. 
Archivos necesarios: sample_libsvm_data.txt, titanic.csv, customer_churn.csv, new_customers.csv,
Script: regresion_logistica_spark.py

## Pendiente ejercicios Python para automatización, RPA, testing (pytest), mobile & web apps (django)
