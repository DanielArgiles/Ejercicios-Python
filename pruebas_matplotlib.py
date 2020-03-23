import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#python -m pip install -U matplotlib

x= np.arange(0,20)
y=x**2
plt.plot(x,y,'r')
plt.show()
plt.plot(x,y,'g*')
plt.title(("Título del gráfico"))
plt.xlabel("eje de los valores x")
plt.ylabel("eje y")
plt.show()

array=np.arange(0,50).reshape(10,5)
print(array)
plt.imshow(array) # cada número del array tiene su color
plt.colorbar()
plt.show()
print("\n")
array2=np.random.randint(0,1000,100)
print(array2)
print("\n")
array2=array2.reshape(10,10) # organiza datos en 10 filas y 10 columnas
print(array2)
plt.imshow(array2)
plt.colorbar()
plt.show()

print("\n")
dataframe= pd.read_csv('personas.csv')
print(dataframe)
dataframe.plot(x='Salario',y='Edad',kind="bar")
plt.show()
dataframe.plot(x='Salario',y='Edad',kind="scatter") 	
plt.show()
