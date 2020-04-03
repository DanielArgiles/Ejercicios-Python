# Ejemplo de Algoritmo de clasificación: A partir de los datos de un fichero de Excel (csv) llamado ingresos.csv, se predicen los ingresos de una persona en función de sus características.

import pandas as pd
import tensorflow as tf
import os 

os.chdir("C:\\Users\\Daniel\\CursoPython-Final\\AprendeMachineLearning")
ingresos= pd.read_csv('ingresos.csv')

ingresos['income'].unique() # nos devuelve un array con los elementos diferentes en la columna 'income'

#(<=50 K) = 0, (>50 k)=1
def cambio_valor(valor):
    if valor == '<=50K':
        return 0
    else:
        return 1
        
ingresos['income']=ingresos['income'].apply(cambio_valor)
print(ingresos.head())

# Divimos los datos, en datos de entrenamiento y datos de pruebas
from sklearn.model_selection import train_test_split

# Eliminamos (drop) de nuestro datafrome la columna 'income'. axis =1 indicamos que lo hacemos sobre columnas, y no filas axis=0
datos_x=ingresos.drop('income',axis=1)
print(datos_x.head())

datos_y=ingresos['income'] # los datos que queremos predecir
print(datos_y)

# Dividir datos x y datos y en 2 conjuntos
x_train,x_test,y_train,y_test=  train_test_split(datos_x,datos_y,test_size=0.3) # 30% datos en test, y 70% en train (tanto para x como y)
print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

# Creamos las variables donde vamos a almacenar los valores de las columnas
print(ingresos.columns)

gender=tf.feature_column.categorical_column_with_vocabulary_list("gender",['Female,Male']) 
# Para columnas de tipo texto (y no numéricas) donde no sabemos el número de elementos que hay. Ponemos un máximo de 1000 elementos
occupation=tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
marital_status=tf.feature_column.categorical_column_with_hash_bucket("marital-status",hash_bucket_size=1000)
relationship=tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=1000)
education=tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
native_country=tf.feature_column.categorical_column_with_hash_bucket("native-country",hash_bucket_size=1000)
workclass=tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=1000)
# Variables de tipo numérico
age=tf.feature_column.numeric_column("age")
fnlwgt=tf.feature_column.numeric_column("fnlwgt")
educational_num=tf.feature_column.numeric_column("educational-num")
capital_gain=tf.feature_column.numeric_column("capital-gain")
capital_loss=tf.feature_column.numeric_column("capital-loss")
hours_per_week=tf.feature_column.numeric_column("hours-per-week")

columnas_categorias=[gender,occupation,marital_status,relationship,education,native_country,workclass,age,fnlwgt,educational_num,capital_gain,capital_loss,hours_per_week]

# Creamos una función de entrada para la estimación, le pasamos los datos de entrenamiento y las soluciones
funcion_entrada = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

# Creamos el modelo de tipo clasificación lineal
modelo=tf.compat.v1.estimator.LinearClassifier(feature_columns= columnas_categorias)

# Entrenamos 8000 veces el modelo creado anteriormente 
modelo.train(input_fn= funcion_entrada , steps=8000)

# Hacemos una predicción de y en función de x
funcion_prediccion= tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=len(x_test),shuffle=False)

generador_predicciones=modelo.predict(input_fn=funcion_prediccion)
predicciones=list(generador_predicciones)
predicciones_finales=[prediccion['class_ids'][0] for prediccion in predicciones]
print(predicciones_finales)

# Generamos un informe para ver el rendimiento de nuestro modelo
from sklearn.metrics import classification_report
print(classification_report(y_test,predicciones_finales))

