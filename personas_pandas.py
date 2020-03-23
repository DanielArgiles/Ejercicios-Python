import pandas as pd 

print("\n Pruebas con archivo formato csv. Delimitado por comas (,)")
dataframe=pd.read_csv('personas.csv')
print(dataframe)
print("\n") # Salto de línea
print(dataframe.describe())
print("\n") # Salto de línea
print(dataframe['Nombre'])
print("\n") # Salto de línea
print(dataframe['Nombre'][0])
print("\n") # Salto de línea
print(dataframe[['Nombre','Apellido']])
print("\n") # Salto de línea
print(dataframe['Salario']>1000)
print("\n") # Salto de línea
filtro= dataframe['Salario']>1000
dataframe2= dataframe[filtro]
print(dataframe2)
print("\n") # Salto de línea
print(dataframe2.values)
print("\n") # Salto de línea
array=dataframe2.values
print(array)
print("\n") # Salto de línea
print(array[0,0])
print("\n") # Salto de línea
print(array[0,1])
