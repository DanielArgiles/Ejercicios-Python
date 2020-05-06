from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('mytree').getOrCreate()
from pyspark.ml import Pipeline
from pyspark.ml.classification import (RandomForestClassifier,GBTClassifier,
                                    DecisionTreeClassifier)
data=spark.read.format('libsvm').load('sample_libsvm_data.txt')
# Vemos una tabla con la columna label y la columna features
data.show()
# Dividimos en datos de entramiento y datos de prueba
train_data,test_data=data.randomSplit([0.7,0.3])
dtc= DecisionTreeClassifier()
rfc= RandomForestClassifier(numTrees=100) # cuantos más árboles agreguemos mayor será el tiempo de cálculo
gbt= GBTClassifier()
# Ajustamos los tres modelos
dtc_model=dtc.fit(train_data)
rfc_model=rfc.fit(train_data)
gbt_model=gbt.fit(train_data)
# Transformamos los datos de prueba para obtener predicciones
dtc_preds= dtc_model.transform(test_data)
rfc_preds= rfc_model.transform(test_data)
gbt_preds= gbt_model.transform(test_data)
# Tenemos las columnas: label|features|rawPrediction|probability|prediction.
# prediction devolverá el actual label
# Para Decision Tree and Random Forest, rawPredictionCol = 'rawPrediction' por defecto
dtc_preds.show()
# Seguimos teniendo las columnas: abel|features|rawPrediction|probability|prediction.
rfc_preds.show()
# Ahora solo tendriamos las columnas : label|features|prediction=GBT, yo he obtenido las mismas que antes
gbt_preds.show()
gbt_preds.printSchema()
# Aunque este es un conjunto de datos de clasificación binario, si solo hago un evaluador de clasificación binario,
# solo podré obtener la curva de recuperación de precisión y AUC
# con el evaluador de clasificación multiclase puedo obtener métricas directas como precisión, recuperación o exactitud
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval= MulticlassClassificationEvaluator(metricName='accuracy')
# Veamos como obtenemos valores poco realistas, muy exactos ya que data contiene información muy fácil de separar
# Obtengo DTC accuracy = 0.92
print('DTC ACCURACY:')
acc_eval.evaluate(dtc_preds)
# Obtengo RFC accuracy = 1
print('RFC ACCURACY:')
acc_eval.evaluate(rfc_preds)
# Obtengo RFC accuracy = 0.92
print('GBT ACCURACY:')
acc_eval.evaluate(gbt_preds)
# Importancia de la característica, algo que será interesante en el proyecto de consultoría
rfc_model.featureImportances

#---Code Along Example---#
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('tree').getOrCreate()
data=spark.read.csv('College.csv',inferSchema=True,header=True)
data.printSchema()
data.head(1)
from pyspark.ml.feature import VectorAssembler
data.columns
assembler= VectorAssembler (inputCols=['Apps',
 'Accept',
 'Enroll',
 'Top10perc',
 'Top25perc',
 'F_Undergrad',
 'P_Undergrad',
 'Outstate',
 'Room_Board',
 'Books',
 'Personal',
 'PhD',
 'Terminal',
 'S_F_Ratio',
 'perc_alumni',
 'Expend',
 'Grad_Rate'],outputCol='features')
output=assembler.transform(data)
# Tenemos que saber si private es 0 o 1
from pyspark.ml.feature import StringIndexer
indexer= StringIndexer(inputCol='Private',outputCol='PrivateIndex')
output_fixed=indexer.fit(output).transform(output)
# Vemos como hemos añadido el vector features y PrivateIndex como double
output_fixed.printSchema()
"""
root
 |-- School: string (nullable = true)
 |-- Private: string (nullable = true)
 |-- Apps: integer (nullable = true)
 |-- Accept: integer (nullable = true)
 |-- Enroll: integer (nullable = true)
 |-- Top10perc: integer (nullable = true)
 |-- Top25perc: integer (nullable = true)
 |-- F_Undergrad: integer (nullable = true)
 |-- P_Undergrad: integer (nullable = true)
 |-- Outstate: integer (nullable = true)
 |-- Room_Board: integer (nullable = true)
 |-- Books: integer (nullable = true)
 |-- Personal: integer (nullable = true)
 |-- PhD: integer (nullable = true)
 |-- Terminal: integer (nullable = true)
 |-- S_F_Ratio: double (nullable = true)
 |-- perc_alumni: integer (nullable = true)
 |-- Expend: integer (nullable = true)
 |-- Grad_Rate: integer (nullable = true)
 |-- features: vector (nullable = true)
 |-- PrivateIndex: double (nullable = false)

"""
final_data=output_fixed.select('features','PrivateIndex')
train_data,test_data=final_data.randomSplit([0.7,0.3])
from pyspark.ml.classification import (DecisionTreeClassifier,GBTClassifier,
                                      RandomForestClassifier)
from pyspark.ml import Pipeline
dtc= DecisionTreeClassifier(labelCol='PrivateIndex',featuresCol='features')
rfc= RandomForestClassifier(labelCol='PrivateIndex',featuresCol='features')
gbt= GBTClassifier(labelCol='PrivateIndex',featuresCol='features') # Gradient Boosted Trees.

dtc_model= dtc.fit(train_data)
rfc_model= rfc.fit(train_data)
gbt_model= gbt.fit(train_data)

dtc_preds= dtc_model.transform(test_data)
rfc_preds= rfc_model.transform(test_data)
gbt_preds=gbt_model.transform(test_data)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
my_binary_eval=BinaryClassificationEvaluator(labelCol='PrivateIndex')
# Obtenemos una evaluación del DTC de 0.92
print('DTC')
print(my_binary_eval.evaluate(dtc_preds))
# Obtenemos una evaluación del RFC de 0.97
print('RFC')
print(my_binary_eval.evaluate(rfc_preds))
# Con GBT no tenemos probability ni prediction
gbt_preds.printSchema()
rfc_preds.printSchema()
my_binary_eval2=BinaryClassificationEvaluator(labelCol='PrivateIndex',
                                             rawPredictionCol='prediction')
# Obtenemos una evaluación del GBT de 0.90
print('GBT')
print(my_binary_eval2.evaluate(gbt_preds))
# Ahora obtenemos una evaluación de RFC mayor, de 0,98, al añadir árboles
rfc= RandomForestClassifier(numTrees=150,labelCol='PrivateIndex',featuresCol='features')
rfc_model= rfc.fit(train_data)
rfc_preds= rfc_model.transform(test_data)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
my_binary_eval=BinaryClassificationEvaluator(labelCol='PrivateIndex')
print('RFC')
print(my_binary_eval.evaluate(rfc_preds))

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval=MulticlassClassificationEvaluator(labelCol='PrivateIndex',
                                           metricName='accuracy')
rfc_acc=acc_eval.evaluate(rfc_preds)
# Obtenemos una exactitud (accuracy) de 0,93
rfc_acc

#---Proyecto de consultoría---#
# Ha sido contratado por una compañía de alimentos para perros en St. Louis, Missouri, para tratar de predecir por qué algunos lotes de alimentos para perros se están echando a perder mucho más rápido de lo previsto.
# La compañía de alimentos para perros primero mezcla un lote de conservantes que contiene 4 químicos conservantes diferentes (A, B, C, D) y luego se completa con un químico "relleno".
# Los científicos de alimentos creen que uno de los conservantes A, B, C o D está causando el problema, ¡pero necesitan su ayuda para descubrir cuál!
# Si bien utilizaremos Machine Learning para resolver esto, no será con su flujo de trabajo dividido típico de entrenamiento / prueba.
"""
Datos:
Pres_A: porcentaje de conservante A en la mezcla
Pres_B: porcentaje de conservante B en la mezcla
Pres_C: porcentaje de conservante C en la mezcla
Pres_D: porcentaje de conservante D en la mezcla
Estropeado: etiqueta que indica si el lote de comida para perros se estropeó o no.
"""

# ¡Mencionamos que estos clasificadores de métodos de árbol tenían un atributo .featureImportances disponible!
# Por lo tanto, podemos crear un modelo, ajustarlo en todos los datos y luego verificar qué característica (conservante) estaba causando el deterioro.
"""
.featureImportances returns:
SparseVector(4, {0: 0.0026, 1: 0.0089, 2: 0.9686, 3: 0.0199})
 Corresponding to a features column:
Row(features=DenseVector([4.0, 2.0, 12.0, 3.0]), Spoiled=1.0)
"""
# Este proyecto de consultoría muestra cómo podemos aplicar el aprendizaje automático de una manera diferente a los ejemplos anteriores (sin división datos de entrenamiento y test).
# Lo que realmente queremos entender es la relación fundamental entre cada columna de características y la propia etiqueta.

from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('tree_consult').getOrCreate()
data= spark.read.csv('dog_food.csv',inferSchema=True,header=True)

# Muestro la fila 1
# [Row(A=4, B=2, C=12.0, D=3, Spoiled=1.0)]
# Tengo los 4 conservantes, y luego si el lote fue o no estropeado (1 si, 0 no)
data.head(1)

from pyspark.ml.feature import VectorAssembler

# ['A', 'B', 'C', 'D', 'Spoiled']
data.columns

assembler=VectorAssembler(inputCols=['A','B','C','D'],outputCol='features')
output=assembler.transform(data)
from pyspark.ml.classification import RandomForestClassifier
rfc=RandomForestClassifier(labelCol='Spoiled',featuresCol='features')
output.printSchema()
"""
root
 |-- A: integer (nullable = true)
 |-- B: integer (nullable = true)
 |-- C: double (nullable = true)
 |-- D: integer (nullable = true)
 |-- Spoiled: double (nullable = true)
 |-- features: vector (nullable = true)
"""
final_data=output.select(['features','Spoiled'])
final_data.show()

# Ahora entrenaremos a nuestro clasificador en los datos reales
rfc_model=rfc.fit(final_data)

# [Row(features=DenseVector([4.0, 2.0, 12.0, 3.0]), Spoiled=1.0)]
# A:(4.0), B:(2.0), C:(12.0), D:(3.0)
final_data.head(1)

# SparseVector(int size, int[] indices, double[] values) 
# SparseVector(4, {0: 0.027, 1: 0.0194, 2: 0.9281, 3: 0.0255})
# A :(0: 0.027), B:(1: 0.0194), C:(2: 0.9281),D:(3: 0.0255)
# Esto quiere decir que la letra C es la característica más importante, el conservante que está causando el deterioro. 
rfc_model.featureImportances

