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

