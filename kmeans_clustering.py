# Clustering
# Ahora tendremos datos sin etiquetar e intentaremos "descubrir" posibles etiquetas, a través de la agrupación o clustering.
# Ingresa algunos datos sin etiquetar, y el algoritmo de aprendizaje no supervisado (No hay training/test data split) devuelve posibles grupos de datos.
# Puede ser difícil evaluar los grupos o clusters para "corrección".
# Una gran parte de poder interpretar los grupos asignados se reduce al conocimiento del dominio
# Muchos problemas de agrupamiento no tienen un enfoque o respuesta 100% correctos, ¡esa es la naturaleza del aprendizaje no supervisado!

""" 
K Means Clustering es un algoritmo de aprendizaje no supervisado que intentará agrupar clústeres similares en sus datos.
El objetivo general es dividir los datos en grupos distintos de modo que las observaciones dentro de cada grupo sean similares
Elija un número de grupos "K"
Asigna aleatoriamente cada punto a un grupo
Hasta que los grupos dejen de cambiar, repita lo siguiente:
Para cada grupo, calcule el centroide del grupo tomando el vector medio de puntos en el grupo
Asigne cada punto de datos al grupo para el cual el centroide es el más cercano.

No hay una respuesta fácil para elegir un "mejor" valor K
Una forma es el método del codo
En primer lugar, calcule la suma del error al cuadrado (SSE) para algunos valores de k (por ejemplo, 2, 4, 6, 8, etc.).
El SSE se define como la suma de la distancia al cuadrado entre cada miembro del grupo y su centroide.
Si traza k contra el SSE, verá que el error disminuye a medida que k aumenta; Esto se debe a que cuando aumenta el número de grupos, deberían ser más pequeños, por lo que la distorsión también es menor.
La idea del método del codo es elegir la k en la cual el SSE disminuye abruptamente.
Esto produce un "efecto codo" en el gráfico.
Pyspark por sí solo no admite un mecanismo de trazado, pero podría usar collect () y luego trazar los resultados con matplotlib u otras bibliotecas de visualización.
"""

#---Clustering Code Example---#
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('cluster').getOrCreate()
from pyspark.ml.clustering import KMeans
dataset=spark.read.format('libsvm').load('sample_kmeans_data.txt')
dataset.show()
"""
dataset-- 6 filas de datos:
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  0.0|           (3,[],[])|
|  1.0|(3,[0,1,2],[0.1,0...|
|  2.0|(3,[0,1,2],[0.2,0...|
|  3.0|(3,[0,1,2],[9.0,9...|
|  4.0|(3,[0,1,2],[9.1,9...|
|  5.0|(3,[0,1,2],[9.2,9...|
+-----+--------------------+
"""
final_data=dataset.select('features')
final_data.show()
"""
final_data solo tiene "features", 5 filas:
+--------------------+
|            features|
+--------------------+
|           (3,[],[])|
|(3,[0,1,2],[0.1,0...|
|(3,[0,1,2],[0.2,0...|
|(3,[0,1,2],[9.0,9...|
|(3,[0,1,2],[9.1,9...|
|(3,[0,1,2],[9.2,9...|
+--------------------+
"""
# Creamos el modelo (K=2)
kmeans= KMeans().setK(2).setSeed(1)
# Ajustamos el modelo
model = kmeans.fit(final_data)
# Calculamos el costo de nuestro conjunto de datos real
wssse= model.computeCost(final_data)
# Obtengo un wssse= 0.11999999999994547
print(wssse)
# Verificamos los centros de clusters
centers=model.clusterCenters()
# Obtengo centers= [array([0.1, 0.1, 0.1]), array([9.1, 9.1, 9.1])]
centers
# Queremos saber a que grupo pertenece cada fila de final_data.
# Es decir, saber la etiqueta que creeé para ello
# Para ello, llamamos al modelo y transformamos, pasándole final_data
# Obtengo : DataFrame[features: vector, prediction: int]
model.transform(final_data)
results= model.transform(final_data)
results.show()
"""
results:
+--------------------+----------+
|            features|prediction|
+--------------------+----------+
|           (3,[],[])|         0|
|(3,[0,1,2],[0.1,0...|         0|
|(3,[0,1,2],[0.2,0...|         0|
|(3,[0,1,2],[9.0,9...|         1|
|(3,[0,1,2],[9.1,9...|         1|
|(3,[0,1,2],[9.2,9...|         1|
+--------------------+----------+

Obtenemos la predicción real, y esto quiere decir que las 3 primeras filas pertenecen al primer grupo y las 3 últimas al segundo grupo.
"""

# Si ahora K=3
kmeans= KMeans().setK(3).setSeed(1)
model = kmeans.fit(final_data)
# wssse = 0,075 , disminuye a medida que aumenta K
wssse= model.computeCost(final_data) 
# centers: [array([9.1, 9.1, 9.1]), array([0.05, 0.05, 0.05]), array([0.2, 0.2, 0.2])]
centers=model.clusterCenters()
centers
model.transform(final_data)
results= model.transform(final_data)
results.show()
"""
results:
+--------------------+----------+
|            features|prediction|
+--------------------+----------+
|           (3,[],[])|         1|
|(3,[0,1,2],[0.1,0...|         1|
|(3,[0,1,2],[0.2,0...|         2|
|(3,[0,1,2],[9.0,9...|         0|
|(3,[0,1,2],[9.1,9...|         0|
|(3,[0,1,2],[9.2,9...|         0|
+--------------------+----------+

Ahora observamos como hay 3 grupos 0,1,2
"""

#---Example Code Along---#
# Trabajaremos a través de un conjunto de datos real que contiene algunos datos sobre tres tipos de semillas distintos.
# ¡Este es un aprendizaje no supervisado!
# ¡Lo que significa que no tenemos las etiquetas originales para realizar algún tipo de prueba

from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('cluster').getOrCreate()
dataset= spark.read.csv('seeds_dataset.csv',header=True,inferSchema=True)
dataset.printSchema()
""" 
Tenemos 3 variedades diferentes de trigo
Y 7 características:
root
 |-- area: double (nullable = true)
 |-- perimeter: double (nullable = true)
 |-- compactness: double (nullable = true)
 |-- length_of_kernel: double (nullable = true)
 |-- width_of_kernel: double (nullable = true)
 |-- asymmetry_coefficient: double (nullable = true)
 |-- length_of_groove: double (nullable = true)
"""

# [Row(area=15.26, perimeter=14.84, compactness=0.871, length_of_kernel=5.763, width_of_kernel=3.312, asymmetry_coefficient=2.221, length_of_groove=5.22)]
dataset.head(1)

# No hay una etiqueta real que indique el tipo de semilla o a que grupos pertenecen, pero si sabemos que hay 3 variedades diferentes de trigo
# Nuestro conocimiento es K=3
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

dataset.columns
"""
['area',
 'perimeter',
 'compactness',
 'length_of_kernel',
 'width_of_kernel',
 'asymmetry_coefficient',
 'length_of_groove']
"""

assembler=VectorAssembler(inputCols=dataset.columns,
                         outputCol='features')
final_data=assembler.transform(dataset)
final_data.printSchema()
"""
root
 |-- area: double (nullable = true)
 |-- perimeter: double (nullable = true)
 |-- compactness: double (nullable = true)
 |-- length_of_kernel: double (nullable = true)
 |-- width_of_kernel: double (nullable = true)
 |-- asymmetry_coefficient: double (nullable = true)
 |-- length_of_groove: double (nullable = true)
 |-- features: vector (nullable = true)
"""

# Escalamos datos
from pyspark.ml.feature import StandardScaler
scaler=StandardScaler(inputCol='features',
                     outputCol='scaledFeatures',
                        )
scaler_model=scaler.fit(final_data)
final_data=scaler_model.transform(final_data)

#[Row(area=15.26, perimeter=14.84, compactness=0.871, length_of_kernel=5.763, width_of_kernel=3.312, asymmetry_coefficient=2.221, length_of_groove=5.22, features=DenseVector([15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]), scaledFeatures=DenseVector([5.2445, 11.3633, 36.8608, 13.0072, 8.7685, 1.4772, 10.621]))]
final_data.head(1)

kmeans= KMeans(featuresCol='scaledFeatures',k=3)
model= kmeans.fit(final_data)

# Obtengo un wssse= 428.6839522475977
print('WSSSE')
print(model.computeCost(final_data))

centers=model.clusterCenters()
# 3 arrays, porque K=3. De 7 dimensiones cada uno.
print(centers)
"""
[array([ 4.9418976 , 10.95423183, 37.31729526, 12.41525595,  8.61430248,
        1.78007698, 10.38784356]), array([ 6.3407095 , 12.39263108, 37.41143125, 13.92892299,  9.77251635,
        2.42396447, 12.28547936]), array([ 4.078007  , 10.15076404, 35.87686106, 11.81860981,  7.5430707 ,
        3.17727834, 10.39174095])]
"""
model.transform(final_data).show()
# Obtengo la predicción para cada grupo. Vemos que son 0,1,2
model.transform(final_data).select('prediction').show()
"""
+----------+
|prediction|
+----------+
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         1|
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         0|
|         2|
+----------+
only showing top 20 rows
"""

#---Consulting Project---#
# ¡Es hora de que vayas a San Francisco para ayudar a una startup tecnológica!
# ¡Han sido pirateados recientemente y necesitan tu ayuda para conocer a los piratas informáticos!
# Afortunadamente, sus ingenieros forenses han obtenido datos valiosos sobre los hacks, incluida información como el tiempo de sesión, las ubicaciones, la velocidad de escritura de wpm, etc.
"""
'Session_Connection_Time': How long the session lasted in minutes
'Bytes Transferred': Number of MB transferred during session
'Kali_Trace_Used': Indicates if the hacker was using Kali Linux
'Servers_Corrupted': Number of server corrupted during the attack
'Pages_Corrupted': Number of pages illegally accessed
'Location': Location attack came from (Probably useless because the hackers used VPNs)
'WPM_Typing_Speed': Their estimated typing speed based

"""
# La empresa de tecnología tiene 3 posibles piratas informáticos que perpetraron el ataque.
# Están seguros de los primeros dos hackers, pero no están muy seguros de si el tercer hacker estuvo involucrado o no.
# Han pedido tu ayuda
# Un último hecho clave, el ingeniero forense sabe que los piratas informáticos intercambian los ataques.
# Lo que significa que cada uno debería tener aproximadamente la misma cantidad de ataques.
# Por ejemplo, si hubo 100 ataques en total, entonces, en una situación de 2 hackers, cada uno debería tener alrededor de 50 hacks, en una situación de tres hackers, cada uno tendría unos 33 hacks.

from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('cluster').getOrCreate()
dataset= spark.read.csv('hack_data.csv',header=True,inferSchema=True)

# Row(Session_Connection_Time=8.0, Bytes Transferred=391.09, Kali_Trace_Used=1, Servers_Corrupted=2.96, Pages_Corrupted=7.0, Location='Slovenia', WPM_Typing_Speed=72.37)
dataset.head()

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
dataset.columns
"""
['Session_Connection_Time',
 'Bytes Transferred',
 'Kali_Trace_Used',
 'Servers_Corrupted',
 'Pages_Corrupted',
 'Location',
 'WPM_Typing_Speed']
"""

# Solo columnas con valores numéricos
feat_cols=['Session_Connection_Time',
 'Bytes Transferred',
 'Kali_Trace_Used',
 'Servers_Corrupted',
 'Pages_Corrupted',
 'WPM_Typing_Speed']

assembler= VectorAssembler(inputCols=feat_cols,outputCol='features')
final_data= assembler.transform(dataset)
final_data.printSchema()
"""
root
 |-- Session_Connection_Time: double (nullable = true)
 |-- Bytes Transferred: double (nullable = true)
 |-- Kali_Trace_Used: integer (nullable = true)
 |-- Servers_Corrupted: double (nullable = true)
 |-- Pages_Corrupted: double (nullable = true)
 |-- Location: string (nullable = true)
 |-- WPM_Typing_Speed: double (nullable = true)
 |-- features: vector (nullable = true)
"""

# Escalamos
from pyspark.ml.feature import StandardScaler
scaler=StandardScaler(inputCol='features',outputCol='scaledFeatures')
scaler_model=scaler.fit(final_data)
cluster_final_data=scaler_model.transform(final_data)
cluster_final_data.printSchema()
"""
root
 |-- Session_Connection_Time: double (nullable = true)
 |-- Bytes Transferred: double (nullable = true)
 |-- Kali_Trace_Used: integer (nullable = true)
 |-- Servers_Corrupted: double (nullable = true)
 |-- Pages_Corrupted: double (nullable = true)
 |-- Location: string (nullable = true)
 |-- WPM_Typing_Speed: double (nullable = true)
 |-- features: vector (nullable = true)
 |-- ScaledFeatures: vector (nullable = true)
"""

kmeans2= KMeans(featuresCol='scaledFeatures',k=2)
kmeans3= KMeans(featuresCol='scaledFeatures',k=3)

model_k2=kmeans2.fit(cluster_final_data)
model_k3=kmeans3.fit(cluster_final_data)

model_k3.transform(cluster_final_data).select('prediction').show()
"""
+----------+
|prediction|
+----------+
|         2|
|         1|
|         2|
|         2|
|         1|
|         2|
|         2|
|         2|
|         2|
|         2|
|         2|
|         1|
|         1|
|         1|
|         2|
|         2|
|         2|
|         1|
|         2|
|         1|
+----------+
only showing top 20 rows
"""

# Tenemos 3 grupos, pero no es una división par. Recuentos desiguales
model_k3.transform(cluster_final_data).groupBy('prediction').count().show()
"""
+----------+-----+
|prediction|count|
+----------+-----+
|         1|   84|
|         2|   83|
|         0|  167|
+----------+-----+
"""

# Ahora tenemos una división exacta e igual a 167 en ambos grupos 0 y 1.
# Por tanto, tendremos 2 hackers.
model_k2.transform(cluster_final_data).groupBy('prediction').count().show()
"""
+----------+-----+
|prediction|count|
+----------+-----+
|         1|  167|
|         0|  167|
+----------+-----+
"""
