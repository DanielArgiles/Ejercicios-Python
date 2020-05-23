"""
Spark Streaming es una extensión de la API central de Spark que permite el procesamiento de flujo en vivo escalable, de alto rendimiento y tolerante a fallas.
Los datos pueden ser ingeridos de muchas fuentes como Kafka, Flume, Kinesis o TCP, y pueden procesarse utilizando algoritmos complejos expresados ​​con funciones de alto nivel como mapear, reducir, unir y ventana.
Internamente, Spark Streaming recibe flujos de datos de entrada en vivo y divide los datos en lotes (batches), que luego son procesados ​​por el motor Spark para generar el flujo final de resultados en lotes.
Primero trabajaremos con un simple ejemplo de streaming.
Necesitará usar simultáneamente el cuaderno jupyter y un terminal para esto.
Esto es más fácil de seguir a través de una instalación local usando Virtual Box.
Después realizaremos un Proyecto de Análisis de Twitter.
Para seguir este proyecto, deberá instalar algunas bibliotecas de visualización y configurar una cuenta de desarrollador de Twitter.
Las diversas fuentes de datos posibles (Kafka, Flume, Kinesis, etc.) no pueden mostrarse de manera realista en una sola configuración de computadora.
Si su lugar de trabajo requiere el uso de una de estas fuentes, Spark proporciona guías de integración completas.
Información adicional sobre Spark Streaming: 
1) http://spark.apache.org/docs/latest/structured-streaming-programming-guide.html
2) http://spark.apache.org/docs/latest/streaming-programming-guide.html
3) http://spark.apache.org/docs/latest/streaming-programming-guide.html#linking
4) http://spark.apache.org/docs/latest/streaming-kafka-integration.html
"""

#-----Spark Streaming Example Code Along-----#

"""

Debido a que usaremos Spark Streaming y no streaming estructurado (aún experimental y en Alpha) necesitamos usar alguna sintaxis "RDD" más antigua.
Esto se debe al uso de un SparkContext en lugar de una SparkSession.
Construiremos una aplicación muy simple que se conecta a un flujo local de datos (un terminal abierto) a través de una conexión de socket.
Luego contará las palabras para cada línea que escribimos.

Los pasos para la transmisión serán:
-Crear un SparkContext
-Crea un StreamingContext
-Cree un Socket Text Stream
-Lea las líneas como un "DStream"


Los pasos para trabajar con los datos:
-Divida la línea de entrada en una lista de palabras.
-Asigna cada palabra a una tupla: (palabra, 1)
-Luego agrupe (reduzca) las tuplas por la palabra (clave) y resuma el segundo argumento (el número uno)

Eso nos proporcionará un recuento de palabras en el formulario ("hola", 3) para cada línea.
Como nota rápida, la sintaxis RDD depende en gran medida de las expresiones lambda, que son funciones rápidas anónimas.
"""

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
sc= SparkContext('local[2]','NetworkWordCount') # 2 hilos locales de trabajo
ssc=StreamingContext(sc,1) #intervalo de 1 segundo
lines= ssc.socketTextStream('localhost',9999)
words=lines.flatMap(lambda line: line.split(' '))
pairs= words.map(lambda word:(word,1))
word_counts= pairs.reduceByKey(lambda num1,num2:num1+num2)
word_counts.pprint()
# Ahora escribimos en consola linux ubuntu un comando netcat para transferir : nc -lk 9999 
# Buscar comando para Anaconda promt
# Instalar netcat en anaconda navigator: conda install -c bkreider/label/main2 netcat o  conda install -c bkreider netcat

# Justo después escribimos en jupyter notebook
ssc.start()
# Y ahora escribimos algo en consola , ejemplo : hello world, y observamos como se verá en jupyter notebook


#-----Twitter Project-----#








