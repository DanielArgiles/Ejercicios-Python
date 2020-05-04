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

