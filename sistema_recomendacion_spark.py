"""
¡Aprendamos cómo construir un sistema de recomendación con Spark y Python!
No hay un proyecto de consultoría o un ejemplo de documentación para esta sección, porque la facilidad de uso de Spark no se presta para ser probado.
Para más interés ver: Sistemas de recomendación de Jannach y Zanker
Los sistemas de recomendación completamente desarrollados e implementados pueden ser complejos e intensivos en recursos.
Dado que los sistemas de recomendación completos requieren un fondo de álgebra lineal pesado, intentaremos proporcionar solo una descripción general de alto nivel en este ejercicio.

Los dos tipos más comunes de sistemas de recomendación son el filtrado basado en contenido y el colaborativo (CF).
El filtrado colaborativo produce recomendaciones basadas en el conocimiento de la actitud de los usuarios hacia los elementos, es decir, utiliza la "sabiduría de la multitud" para recomendar elementos.
Los sistemas de recomendación basados en contenido se centran en los atributos de los elementos y le dan recomendaciones basadas en la similitud entre ellos.

En general, el filtrado colaborativo (CF) se usa más comúnmente que los sistemas basados en contenido porque generalmente brinda mejores resultados y es relativamente fácil de entender (desde una perspectiva de implementación general).
El algoritmo tiene la capacidad de realizar funciones de aprendizaje por sí solo, lo que significa que puede comenzar a aprender por sí mismo qué funciones utilizar.
Estas técnicas tienen como objetivo completar las entradas faltantes de una matriz de asociación de elementos de usuario.
spark.ml actualmente admite el filtrado colaborativo basado en modelos, en el que los usuarios y productos se describen mediante un pequeño conjunto de factores latentes que se pueden usar para predecir entradas faltantes.
spark.ml usa el algoritmo de mínimos cuadrados alternos (ALS) para aprender estos factores latentes.
¡Sus datos deben estar en un formato específico para funcionar con el Algoritmo de recomendación ALS de Spark!
ALS es básicamente un enfoque de Factorización de matriz para implementar un algoritmo de recomendación que descompone su matriz de usuario / elemento grande en factores de usuario de menor dimensión y factores de elemento.

La comprensión intuitiva de un sistema de recomendación colaborativo es la siguiente:
Imagina que tenemos 3 clientes: 1,2,3.
También tenemos algunas películas: A, B, C
¡Los clientes 1 y 2 realmente disfrutan las películas A y B y las califican como cinco de cinco estrellas!
# 1 y # 2 no les gusta la película C, y le dan una calificación de una estrella.
Ahora tenemos un nuevo cliente n. ° 3, que informa una reseña de 5 estrellas para la película A.
¿Qué nueva película deberíamos recomendar, B o C?
Bueno, basado en el filtrado colaborativo, recomendamos la película B, porque los usuarios n. ° 1 y n. ° 2 también disfrutaron eso (y la película A)

Un sistema basado en contenido no necesitaría tener en cuenta a los usuarios.
Simplemente agruparía películas en función de las características (duración, género, actores, etc.)
A menudo, los sistemas de recomendación reales tienen combinaciones de métodos.

"""

from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('rec').getOrCreate()
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
data= spark.read.csv('movielens_ratings.csv',inferSchema=True,header=True)
data.show()
"""
+-------+------+------+
|movieId|rating|userId|
+-------+------+------+
|      2|   3.0|     0|
|      3|   1.0|     0|
|      5|   2.0|     0|
|      9|   4.0|     0|
|     11|   1.0|     0|
|     12|   2.0|     0|
|     15|   1.0|     0|
|     17|   1.0|     0|
|     19|   1.0|     0|
|     21|   1.0|     0|
|     23|   1.0|     0|
|     26|   3.0|     0|
|     27|   1.0|     0|
|     28|   1.0|     0|
|     29|   1.0|     0|
|     30|   1.0|     0|
|     31|   1.0|     0|
|     34|   1.0|     0|
|     37|   1.0|     0|
|     41|   2.0|     0|
+-------+------+------+
only showing top 20 rows
"""

data.describe().show()
# Observamos que tenemos 1501 entradas de datos o filas.
# 30 usuarios (del 0 al 29)
# 5 Valores de ranting (del 1.0 al 5.0)
# 100 movieId (del 0 al 99)
"""
+-------+------------------+------------------+------------------+
|summary|           movieId|            rating|            userId|
+-------+------------------+------------------+------------------+
|  count|              1501|              1501|              1501|
|   mean| 49.40572951365756|1.7741505662891406|14.383744170552964|
| stddev|28.937034065088994| 1.187276166124803| 8.591040424293272|
|    min|                 0|               1.0|                 0|
|    max|                99|               5.0|                29|
+-------+------------------+------------------+------------------+
"""

training,test= data.randomSplit([0.8,0.2])
als= ALS(maxIter=5,regParam=0.01,userCol='userId',itemCol='movieId',ratingCol='rating')
model= als.fit(training)
predictions= model.transform(test)
predictions.show()
"""
+-------+------+------+----------+
|movieId|rating|userId|prediction|
+-------+------+------+----------+
|     31|   1.0|    26| -2.104602|
|     31|   1.0|    27|  1.552695|
|     31|   4.0|    12|-0.5834996|
|     31|   1.0|    13|0.31738144|
|     31|   1.0|    19|0.10276699|
|     31|   1.0|    29|0.12467331|
|     31|   3.0|    14| 0.7298139|
|     85|   1.0|    26|-3.0117252|
|     85|   1.0|    25| 2.7123184|
|     65|   1.0|    28| 0.9381701|
|     65|   2.0|    15| 0.6194817|
|     53|   1.0|     6| 3.1395166|
|     53|   1.0|    23|  1.288641|
|     53|   1.0|     7| 3.7092607|
|     78|   1.0|    28| 0.4855252|
|     78|   1.0|    12|0.45514816|
|     78|   1.0|     8|0.41600037|
|     78|   1.0|    24|0.76659226|
|     34|   3.0|    25| 4.0282216|
|     34|   1.0|    14| 1.1292068|
+-------+------+------+----------+
only showing top 20 rows
"""

evaluator= RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')
rmse=evaluator.evaluate(predictions)

# Obtengo : RMSE = 1.842299496840756
print('RMSE')
print(rmse)

# Veamos todas las películas que ha visto el usuario 11
single_user= test.filter(test['userId']==11).select(['movieId','userId'])
single_user.show()
"""
+-------+------+
|movieId|userId|
+-------+------+
|     10|    11|
|     20|    11|
|     21|    11|
|     25|    11|
|     27|    11|
|     38|    11|
|     39|    11|
|     41|    11|
|     43|    11|
|     67|    11|
|     71|    11|
|     72|    11|
|     79|    11|
+-------+------+

"""

recommendations= model.transform(single_user)
recommendations.orderBy('prediction',ascending=False).show()
"""
+-------+------+----------+
|movieId|userId|prediction|
+-------+------+----------+
|     27|    11| 4.1031857|
|     79|    11| 2.7133596|
|     38|    11| 2.4915576|
|     20|    11| 2.3264537|
|     71|    11| 2.2449198|
|     10|    11|  2.219851|
|     67|    11| 2.1680481|
|     72|    11|  2.139616|
|     43|    11| 2.1187408|
|     21|    11| 1.7894349|
|     39|    11|0.76448506|
|     25|    11|0.67065537|
|     41|    11|0.48817042|
+-------+------+----------+
"""

# El Arranque en frío es un problema de los sistemas de recomendación.
# ¿Qué hacer con los usuarios nuevos en tu plataforma que no han visto ninguna película?
# Una solución es darles una encuesta rápida sobre las películas que han visto
# ¿Puedes clasificarlos rápidamente para nosotros?
# ¿Eres similar al usuario X, Y o Z? 




