import pandas as pd 
 
xlsx=pd.ExcelFile("clientes.xlsx") 

# Imprime los nombres de las hojas del archivo xlsk
print(xlsx.sheet_names)
print("\n") # Salto de línea

#Transformamos en un DataFrame
df1= xlsx.parse("Hoja1")
df2= xlsx.parse("Hoja2")

print(df1)  # Imprime el DataFrame de la Hoja 1
print("\n") # Salto de línea
print(df2) # Imprime el DataFrame de la Hoja 2
print("\n") # Salto de línea

indice=0
for primerafila in df1:
	print([primerafila[0]]) # Imprime letra 0 de la cabecera de DataFrame de Hoja1
	print([primerafila]) # Imprime la cabecera del DataFrame de Hoja1
	indice+=1

print("\n PRUEBAS CON ILOC") 
#El método iloc se utiliza en los DataFrames para seleccionar los elementos en base a su ubicación.
print(df1.iloc[0]) # Imprime Primera fila
print("\n") # Salto de línea
print (df1.iloc[1]) # Imprime Segunda fila
print("\n") # Salto de línea
print (df1.iloc[-1]) # Imprime Última fila
print("\n") # Salto de línea
print(df1.iloc[:, 0]) # Imprime Primera columna
print("\n") # Salto de línea
print(df1.iloc[:, 1]) # Imprime  Segunda columna
print("\n") # Salto de línea
print(df1.iloc[:, -1]) # Imprime Última columna
print("\n") # Salto de línea
print(df1.iloc[0:2]) # Imprime Primeras dos filas
print("\n") # Salto de línea
print(df1.iloc[:, 0:2]) # Imprime Primeras dos columnas
print("\n") # Salto de línea
print(df1.iloc[[0,2,1]])  # Imrpime Primera, tercera y segunda filas
print("\n") # Salto de línea
print(df1.iloc[:, [0,2,1]])  # Imprime Primera, tercera y segunda columnas

print("\n PRUEBAS CON LOC") 
#El método loc se puede usar de dos formas diferentes: seleccionar filas o columnas en base a una etiqueta o seleccionar filas o columnas en base a una condición.
# las filas tienen como etiquete el número de fila. 
#Por lo que el uso de loc parece similar al de iloc para las filas, aunque tiene algunas diferencias.
#En el caso de las columnas se nota más la diferencia ya que se puede acceder a ellas por nombre, tanto sea una como varias.

# Seleccionar Filas en base a su etiqueta
df_sub = df1.loc[1:4] # Selección de filas de la 1 a la 3
print(df_sub.loc[1]) # loc, en base a etiqueta
print("\n") # Salto de línea
print(df_sub.iloc[1]) # compaaro con iloc,  en base a ubicación. índice 1 es fila 2
print("\n") # Salto de línea

# Seleccionar Columnas en base a su etiqueta
print(df1.loc[:, ['Teléfono']])
print("\n") # Salto de línea
print(df1.loc[:, ['Teléfono', 'Población']]) # Varias columnas
print("\n") # Salto de línea

#Seleccionar filas o columnas en base a una condición con loc
is_madrid = df1.loc[:, 'Población'] == 'Madrid'
madrid = df1.loc[is_madrid]
print(madrid) # Imprime solo clientes con población Madrid
print("\n") # Salto de línea
print(madrid.head()) # Con la función head() podemos ver las primeras filas de los datos que hemos cargado
print("\n") # Salto de línea

# Modificación de datos de la columna 0 (Cliente). Se modifica en el programa, no en el archivo xslx
df1.iloc[:, 0]= ['Pepe','Julio','Jesús','Antonio']
print(df1.iloc[:, 0])
print("\n") # Salto de línea
print(df1)  # Imprime el dataframe de la Hoja 1 con los datos modificados
print("\n") # Salto de línea

