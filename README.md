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

## Ejercicio 5 : Lectura de Hoja de Cálculo Excel (xlsx) con Pandas
En este ejercicio se realiza algunas pruebas de selección e impresión de las diferentes filas y columnas de un Libro de Excel (clientes.xlsx), haciendo uso de la librería Pandas.
Inicialmente se transforma cada Hoja del Libro de Excel en un DataFrame, para después leer los datos con los métodos iloc y loc.
Script: clientes_pandas.py.

## Ejercicio 6 : Lectura de un fichero de Excel (csv) con Pandas 
En este ejercicio se realiza algunas pruebas de tratamiento de datos de un fichero csv (valores separados por comas) con el nombre personas.csv, mediante la librería Pandas.
Script: personas_pandas.py

## Ejercicio 7 : Pruebas con NumPy
En este ejercicio se realiza algunas pruebas básicas con NumPy.
Script: pruebas_numpy.py

## Ejercicio 8 : Pruebas con Matplotlib (Visualización de datos)
En este ejercicio se realiza algunas pruebas de visualización de datos.
Script: pruebas_matplotlib.py

## Ejercicio 9 : Pruebas con Scikit Learn
En este ejercicio se realiza unas pruebas para dividir un conjunto de datos disponible en datos para entrenamiento y en datos para test.
Script: pruebas_scikitlearn.py

## Ejercicio 10 : Detector de colores de una imagen con Pandas (Data Science) y OpenCV (Visión Artificial)
En este ejercicio se construye una aplicación a través de la cual se puede obtener automáticamente el nombre del color haciendo doble clic en las diferentes zonas de una imagen (color_animales.jpg).
Nombre del script: detector_colores.py.
Tenemos un archivo de datos (colores.csv) que incluye 865 nombres de colores junto con sus valores RGB y hexadecimales, sabiendo que los colores se componen de 3 colores primarios; rojo, verde y azul (RGB). 
Para obtener el nombre del color correctamente, calculamos una distancia (d) que nos dice como de cerca estamos del color y elegimos el que tiene la distancia mínima.
d = abs (Rojo - ithRedColor) + (Verde - ithGreenColor) + (Azul - ithBlueColor).

## Ejercicio 11 : Pruebas con TensorFlow
En este ejercicio se realiza algunas pruebas de sintáxis básica de TensorFlow.
Script: pruebas_tensorflow.py

## Ejercicio 12 : Mi primera red neuronal con TensorFlow
En este ejercicio se realiza un ejemplo de red neuronal: Z=W*X+B. 
Script: primera_red_neuronal.py

## Ejercicio 13 : Regresión lineal simple con TensorFlow
En este ejercicio se realiza un ejemplo de regresión lineal simple: y=mx+b.
Script: linea_regresion_tensorflow.py
