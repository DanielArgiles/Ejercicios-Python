
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
# pip install -U scikit-learn

datos= np.random.randint(0,100,(100,4)) # 100 filas x 4 columnas, de números enteros aleatorios de 0 a 100
print(datos)
dataframe=pd.DataFrame(data=datos,columns=['c1','c2','c3','etiqueta'])
print("\n")
print(dataframe)

print("\nDividir el conjunto de datos disponible , en datos para entrenamiento y en datos para test")
x=dataframe[['c1','c2','c3']]
print(x)
print("\n")
y=dataframe['etiqueta']
print(y)
print("\n Ahora dividimos cada variable , x e y, una parte para entrenamiento y otra para pruebas")

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3) # 0.3 es el 30% de todos los datos. 30 filas para test y 70 para entrenamiento

print("Train")
print (x_train.shape)
print (y_train.shape)

print("Test")
print (x_test.shape)
print (y_test.shape)
