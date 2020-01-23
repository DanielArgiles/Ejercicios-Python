import numpy as np

print("Creación de un Array de Numpy a partir de una lista o tupla")
np_array = np.array((1,2,3,4,5)) # Crear Array de Numpy a partir de una tupla
np_array = np.array([1,2,3,4,5]) # Crear Array de Numpy a partir de una lista
type(np_array) # comprobar el tipo de dato como array de Numpy
np_array.dtype # Comprobar tipo de dato de los elementos del array de Numpy

np_array_1 = np.array((1,2,3,4,5), dtype=np.float32) # Indicar tipo de datos de los elementos del array de Numpy. En este caso, reales de 32 bits.
np_array_2 = np.array([[1,2],[3,4]]) # crear array de numpy bidimensional
np_array_3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3,3) # Crera matriz como array Numpy

print(np_array_1)
print("\n")
print(np_array_2)
print("\n")
print(np_array_3)

print("\nEncontrar la posición de un elemento en un Array de Numpy")
# La función np.where() devuelve una tupla con la posición del los elementos diferentes de cero de objetos np.array u otros que puedan contener un vector como listas o tuplas. En el caso de que el tipo de dato del objeto sea lógico devuelve la posición de los elementos verdaderos.
np_arrayy = np.array((True, False, True, False))
print(np.where(np_arrayy))

# Encontrar la posición de un elemento en un Array de Numpy. Si el elemento no existe, se obtiene una tupla vacía.
print("\n")
print(np.where(np_array == 5))

# Obtención de los elementos en una matriz
print("\n")
print(np.where(np_array_3 == 7))

# Otros usos: buscar valor máximo
print("\n")
np_array_4 = np.array((1,2,7,3,4,5))
print(np.where(max(np_array_4) == np_array_4))

print("\nSeleccionar elementos en un Array de Numpy")
# Seleccionar un único elemento en un Array de Numpy
array = np.arange(1, 29, 3) # creo un array de 1 a 29 con intervalos de 3 
print(array)
print(array[1])

# Seleccionar una parte de un Array de Numpy (Slicing)
print("\n")
print(array[2:6]) # selección del segundo al quinto elemento (6-1)
print(array[:5])  # selección del 0 al cuarto elemento
print(array[5:])  # selección del 5 al final

# Seleccionar los elementos en una matriz
print("\n")
array_1 = np.arange(1, 10).reshape(3,3) # Creo una matriz convalores del 1 al 10. 3 filas y 3 columnas 
print(array_1)
print(array_1[0,1]) # Imprime elemento. Seleección en primer lugar fila 0, y posición 1
print(array_1[1, :]) # Imprime segunda fila de la matriz

print("\nAplicar una función sobre una fila o columna de una Array de Numpy")
# La función np.apply_along_axis() permite aplicar una función sobre una de las dimisiones de un Array de Numpy
# np.apply_along_axis() acepta como entrada cualquier tipo de objeto que se puede convertir a un Array de Numpy, realizando la conversión de tipos internamente.
matrix = np.arange(1, 10).reshape(3,3)
print(matrix)
print(np.apply_along_axis(sum, 0, matrix)) # Suma de columnas
print(np.apply_along_axis(sum, 1, matrix)) # Suma de filas 
print(np.apply_along_axis(sum, 1, [[1,2],[2,3]])) # Suma elementos de las listas

print("\nInicialización de arrays en Numpy")
# Inicialización de arrays con ceros con np.zeros()
# Forma : np.zeros(shape, dtype=float, order='C')
print(np.zeros(3)) # Vector de ceros
print(np.zeros((3, 2))) # Matriz de ceros

# Inicialización de arrays con unos con np.ones()
print("\n")
print(np.ones(3))
print(np.ones((3, 2)))

# Inicialización de arrays con otros valores
print("\n")
print(np.ones(3) * 3)

print("\nEl método numpy.where()")
# Uso básico de numpy.where(), con el que se puede seleccionar elementos en base a una condición. 
# Si la condición para el elemento i es cierta se selecciona el elemento correspondiente del primer vector, en caso contrario del segundo
print(np.where([True, False, True, False], [1, 2, 3, 4], [5, 6, 7, 8]))

data_1 = np.array([1, 3, 2, 1])
data_2 = np.array([3, 2, 1, 3])
print(np.where(data_1 > data_2, data_1, data_2))
print(np.where(data_1 > 1, data_1, 2))

# Uso avanzado de numpy.where()
print("\n")
data_1 = np.array([9, 17, 15, 20])
data_2 = np.array([20, 5, 13, 18])
print(np.where((data_1 > 10) & (data_1 < 15), data_1, data_2))

print("\nLocalizar valores únicos en arrays Numpy")
# La función unique() permite localizar valores únicos en arrays, la cual se puede utilizar de la siguiente forma:
# np.unique(arr, return_index=False, return_inverse=False, return_counts=False, axis=None)
arr = np.array([12, 12, 9, 1, 2, 9, 8, 2, 1])
print(np.unique(arr))

# Identificar la posición de primera aparición de los valores en arrays Numpy
print(np.unique(arr, return_index=True))
unique, indices = np.unique(arr, return_index=True)
print(arr[indices]) # recuperar los valores únicos a partir del vector original

# Contar el número de apariciones
print("\n")
print(np.unique(arr, return_counts=True))

# Obtener un vector con que construir el original
print("\n")
unique, indices = np.unique(arr, return_inverse=True)
print(unique[indices])

print("\nValores mínimos y máximos en arrays Numpy")
# Las funciones amin() y amax(). Obtener los valores mínimos y/o máximos de un array de Numpy
# np.amin(arr, axis=None, out=None, initial=<no value>)
arr = np.array([12, 7, 6, 11, 2, 9, 15, 5, 14])
print(np.amin(arr)) # Valor mínimo : 2
print(np.amax(arr)) # Valor máximo : 15

# Limitar la búsqueda
print("\n")
print(np.amin(arr, initial=1)) # Se obtiene 1
print(np.amin(arr, initial=9)) # Se obtiene 2
print(np.amax(arr, initial=10)) # Se obtiene 15
print(np.amax(arr, initial=20)) # Se obtiene 20

# Obtener la posición de los elementos
print("\n")
print(np.where(arr == np.amin(arr))) # Se obtiene 4
print(np.where(arr == np.amax(arr))) # Se obtiene 6



print("\nEliminar elementos en arrays de Numpy")
arr = np.array([1,2,3,4,5])
print(np.delete(arr, 1)) # Eliminar los elementos de un vector

# Eliminar las filas o columnas de una matriz
print("\n")
mat = np.array([[11,12,13], [21,22,23], [31,32,33]])
print(mat)
print(np.delete(mat, 1, axis=0)) # elimina segunda fila (indice 1)
print(np.delete(mat, 1, axis=1)) # elimina segunda columna (indice 1)
print(np.delete(mat, (0,2), axis=1)) # Elimina columnas 0 y 2

# Eliminar los valores de una matriz tratanto ésta como un vector
print("\n")
print(np.delete(mat, 5))


print("\nSeleccionar filas y columnas en matrices Numpy")
matrix = np.array(([11, 12, 13], [21, 22, 23], [31, 32, 33])) 
print(matrix[2][1]) # Selecciona un Elemento
print(matrix[2]) # Selecciona Fila 2
print(matrix[0:2]) # Más de una fila (de la 0 a la 2-1)
print(matrix[:, 1]) # Columna 
print(matrix[:, :2]) # Más de una columna 
print(matrix[1:, :2]) # Selección de submatrices


print("\nCrear vectores con valores equiespaciados en Numpy")
print(np.arange(10, 51, 10))
print(np.arange(10, 16))
print(np.arange(10))

 
print("\nSeleccionar elementos condicionalmente en Numpy")
arr = np.arange(10)
print(arr)
menor_5 = arr < 5
print(menor_5)
print(arr[menor_5])
print(arr[(np.mod(arr, 2) == 0) & (arr > 5)])


print("\nañadir elementos en arrays de Numpy con np.append()")
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(np.append(arr, 10))
print(np.append(arr, [11, 12])) #A gregar otro array
mat = np.array([[1, 2, 3],[4, 5, 6]])
print(mat)
print(np.append(mat, [11, 12])) # Agregar elementos en matrices, se agrega en forma de array
print(np.append(mat, [[11, 12, 13]], axis=0)) # Agregar nueva fila 
print(np.append(mat, [[11], [12]], axis=1)) # Agregar nueva columna 


print("\ninicializar arrays de Numpy con un valor")
print(np.full(5, 0))
print(np.full(5, 10))
print(np.full((3, 2), 0)) # 3 filas y 2 columnas con todo 0
print(np.full((4, 4), 10)) # 4 filas y 4 columnas con todo 10
print(np.full((2, 3, 4), 5)) # Crear objetos de tres o más dimisiones. 2 matrices, 3 filas, 4 columnas, todo 5


print("\nInvertir arrays de Numpy")
arr = np.array([1,2,3,4,5])
print(arr)
print(arr[::-1])
print(np.flip(arr)) # otra forma 
arr2D = np.array([[11, 12, 13], 
                  [21, 22, 23],
                  [31, 32, 33]])
print(arr2D)
print(arr2D[::-1]) # Invertir matrices en Numpy
#print(np.flip(arr2D)) # otra forma
print(arr2D[:,::-1]) #Invertir columnas
print(np.flip(arr2D, axis=1)) # Invertir por filas o columnas. 0 es para filas, 1 para columnas
arr2D[:, 1] = arr2D[::-1, 1] # Invertir solamente una fila o columna
arr2D[1] = arr2D[1, ::-1] #  invertir una sola columna (columna 1)
print("")
print(arr2D)
