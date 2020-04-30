from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('mylogreg').getOrCreate()
from pyspark.ml.classification import LogisticRegression
my_data=spark.read.format('libsvm').load('sample_libsvm_data.txt')
my_data.show()
# Creamos el modelo de regresión logística
my_log_reg_model= LogisticRegression()
fitted_logreg_model= my_log_reg_model.fit(my_data)
log_summary=fitted_logreg_model.summary
log_summary.predictions.printSchema()
log_summary.predictions
# Comprobamos si la etiqueta label coincide con la etiqueta prediction(lo que el modelo predijo), si coincide es buena señal.
log_summary.predictions.show()
# Tomamos todos nuestros datos y los dividimos al azar
lr_train,lr_test=my_data.randomSplit([0.7,0.3])
# Creamos el modelo de regresión logística
final_model=LogisticRegression()
final_model.fit(lr_train)
# Ajustamos el modelo en sus datos de entramiento 
fit_final=final_model.fit(lr_train)
#Usamos ese modelo de ajuste que puede evaluar en el conjunto de datos de prueba
prediction_and_labels=fit_final.evaluate(lr_test)
# Mismas columnas que con log_summary.predictions.show() pero:
# Estamos en los datos de prueba por lo que es posible que no obtenga la etiqueta perfecta y la predicción coincidente
prediction_and_labels.predictions.show()
# Exploramos la valoración de la etiqueta predictions del dataframe creando un objeto evaluador 
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                  MulticlassClassificationEvaluator)
# ver documentación:
# http://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#binary-classification
# http://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#multiclass-classification
# Evaluador de clasificación binaria 
my_eval= BinaryClassificationEvaluator()
my_final_roc= my_eval.evaluate(prediction_and_labels.predictions)
# ROC=1 es ajuste perfecto, no muy realista
my_final_roc
# Great example from Databricks: https://docs.databricks.com/applications/machine-learning/mllib/binary-classification-mllib-pipelines.html
# Explanation of AUC: https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it


#------#
# Titanic Dataset, ejercicio muy común para clasificación en Machine Learning
# Se predicen cuantos pasajeros sobrevivieron en función de sus características (edad, cabina, hijos, etc)

"""
Lo realizaremos desde el Notebook de Databricks:
Databricks es un espacio de trabajo, Workspace,  basado en Apache Spark, 
que permite colaborar a científicos de datos y ingenieros de datos en diferentes clusters 
mediante el desarrollo de Notebooks y bajo el soporte de un Runtime basado en Apache Spark donde se ejecutan todas las operaciones.

- Vamos a : https://databricks.com/try-databricks
- Community Edition: for students and educational institutions
- New cluster: myfirstcluster
- New notebook: myfirstnotebook
- Data: Create new table (mytable): drop file titanic.csv, name titanic_csv
- Create table with UI, select cluster, first row is header
- En notebook, abrimos myfirstnotebook y escribimos: 
import pyspark, df=sqlContext.sql("SELECT* FROM titanic_csv"), df.show()
"""

df=spark.sql("SELECT*FROM titanic_csv")
df.printSchema()
"""
root
 |-- PassengerId: integer (nullable = true)
 |-- Survived: integer (nullable = true)
 |-- Pclass: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- SibSp: integer (nullable = true)
 |-- Parch: integer (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: float (nullable = true)
 |-- Cabin: string (nullable = true)
 |-- Embarked: string (nullable = true)
"""
df.columns
my_cols=df.select(['Survived',
   'Pclass','Sex',
   'Age',
   'SibSp',
   'Parch','Fare','Embarked'])
my_final_data=my_cols.na.drop()
from pyspark.ml.feature import(VectorAssembler,VectorIndexer,
                              OneHotEncoder,StringIndexer)
# Indexer asigna un número a cada columna string (ej: A B C será  0 1 2)
gender_indexer= StringIndexer(inputCol='Sex',outputCol='SexIndex')
#ONE HOT ENCODE
# KEY A B C 
# Example A
# [1, 0, 0], false, true, true
gender_encoder= OneHotEncoder(inputCol='SexIndex',outputCol='SexVec')
embark_indexer=StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder=OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')
assembler=VectorAssembler(inputCols= ['Pclass','SexVec','EmbarkVec','Age','SibSp','Parch','Fare'],
                         outputCol='features')
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
log_reg_titanic= LogisticRegression(featuresCol='features',labelCol='Survived')
pipeline= Pipeline(stages=[gender_indexer, embark_indexer,
                          gender_encoder,embark_encoder,
                          assembler,log_reg_titanic])
train_data,test_data=my_final_data.randomSplit([0.7,0.3])
fit_model=pipeline.fit(train_data)
results=fit_model.transform(test_data)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
my_eval=BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Survived')
results.select('Survived','prediction').show()
AUC= my_eval.evaluate(results)
# Obtengo un valor de AUC (Area Under the Curve)=0,81
# AUC se utiliza en el análisis de clasificación para determinar cuál de los modelos utilizados predice mejor las clases. 
AUC
