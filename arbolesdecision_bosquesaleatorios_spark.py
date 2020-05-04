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
