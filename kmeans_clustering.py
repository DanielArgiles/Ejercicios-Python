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
