# Ejemplo de un algoritmo de regresión: A partir de los datos de un fichero de Excel (csv) llamado precios_casas.csv, se predice el valor medio de una casa en función de sus características(habitaciones, latitud, longitud, etc).

import pandas as pd
import tensorflow as tf

casas=pd.read_csv("precios_casas.csv")
print(casas.head())

# Eliminamos (drop) de nuestro datafrome la columna 'median_house_value'. axis =1 indicamos que lo hacemos sobre columnas, y no filas axis=0
casas_x=casas.drop('median_house_value',axis=1)
print(casas_x.head()) # con head() visualizo las 5 primeras filas

casas_y=casas['median_house_value']
print(casas_y.head()) # con head() visualizo las 5 primeras filas

# Dividimos en datos de pruebas y datos de entrenamiento (tanto x como y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(casas_x,casas_y, test_size=0.3)# 30% datos en test, y 70% en train (tanto para x como y)
print(x_test.head())
print(x_train.head())
print(y_test.head())
print(y_train.head())

# Normalizamos (entre 0 y 1) los datos de test y de entrenamiento
from sklearn.preprocessing import MinMaxScaler
normalizador=MinMaxScaler()
normalizador.fit(x_train)
x_train=pd.DataFrame(data=normalizador.transform(x_train),columns=x_train.columns, index=x_train.index)
x_test=pd.DataFrame(data=normalizador.transform(x_test),columns=x_test.columns,index=x_test.index)

# Creamos las variables con las columnas de categorías (todas las columnas, menos la última que es la objetivo)
print(casas.columns)

longitude=tf.feature_column.numeric_column('longitude')
latitude=tf.feature_column.numeric_column('latitude')
housing_median_age=tf.feature_column.numeric_column('housing_median_age')
total_rooms=tf.feature_column.numeric_column('total_rooms')
total_bedrooms=tf.feature_column.numeric_column('total_bedrooms')
population=tf.feature_column.numeric_column('population')
households=tf.feature_column.numeric_column('households')
median_income=tf.feature_column.numeric_column('median_income')

# Creamos la lista columnas con las variables
columnas=[longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income]

# Creamos una función de entrada para la estimación, le pasamos los datos de entrenamiento y las soluciones
funcion_entrada=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

# Creamos el modelo de tipo regresión
modelo=tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=columnas) # 3 capas , cada una con 10 nodos

# Entrenamos 8000 veces el modelo creado anteriormente 
modelo.train(input_fn=funcion_entrada,steps=8000)

# Hacemos una predicción de y en función de x
funcion_entrada_prediccion=tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1,shuffle=False)

generador_predicciones=modelo.predict(funcion_entrada_prediccion)

predicciones=list(generador_predicciones)
print(predicciones)

# Recogemos los valores de la predicción. 
predicciones_finales=[]
for prediccion in predicciones:
    predicciones_finales.append(prediccion['predictions'])
    
print(predicciones_finales)

# Vemos el rendimiento de la estimación que hemos realizado sobre los precios de las casas
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,predicciones_finales)**0.5) # Error cuadrático medio de la estimación 
