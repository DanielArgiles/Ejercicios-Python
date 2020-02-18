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

## Ejercicio 5 : Lectura de un Libro de Excel en Pandas con los métodos iloc y loc
En este ejercicio se realiza algunas pruebas de selección e impresión de las diferentes filas y columnas de un Libro de Excel (clientes.xlsx), haciendo uso de la librería Pandas.
Inicialmente se transforma cada Hoja del Libro de Excel en un DataFrame, para después leer los datos con los métodos iloc y loc.
Nombre del script: clientes_pandas.py.

## Ejercicio 5 : Numpy fundamental
Script(numpy_fundamental.py) con tareas fundamentales que se pueden realizar con la librería Numpy.

## Ejercicio 6 : Detector de colores de una imagen con Pandas y OpenCV
En este ejercicio se construye una aplicación a través de la cual se puede obtener automáticamente el nombre del color haciendo doble clic en las diferentes zonas de una imagen (colorpic.jpg).
Nombre del script con código fuente :color_detection.py.
Tenemos un archivo de datos (colors.csv) que incluye 865 nombres de colores junto con sus valores RGB y hexadecimales, sabiendo que los colores se componen de 3 colores primarios; rojo, amarillo y azul. 
Para obtener el nombre del color correctamente, calculamos una distancia (d) que nos dice como de cerca estamos del color y elegimos el que tiene la distancia mínima.
d = abs (Rojo - ithRedColor) + (Amarillo - ithYellowColor) + (Azul - ithBlueColor).
