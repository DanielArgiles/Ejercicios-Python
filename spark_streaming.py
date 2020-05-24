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
# Creamos un DStream que se conectará al nombre de host: puerto, como localhost: 9999
# ¡Los firewalls pueden bloquear esto!
lines= ssc.socketTextStream('localhost',9999)
# Dividimos cada línea en palabras
words=lines.flatMap(lambda line: line.split(' '))
# Cuenta cada palabra en cada lote
pairs= words.map(lambda word:(word,1))
word_counts= pairs.reduceByKey(lambda num1,num2:num1+num2)
# Imprimimos los primeros diez elementos de cada RDD generado en este DStream en la consola
word_counts.pprint()
# Ahora escribimos en consola linux ubuntu un comando netcat para transferir : nc -lk 9999 
# Buscar comando para Anaconda promt
# Instalar netcat en anaconda navigator: conda install -c bkreider/label/main2 netcat o  conda install -c bkreider netcat

# Justo después escribimos en jupyter notebook
ssc.start()
# Y ahora escribimos algo en consola , ejemplo : hello world, y observamos como se verá en jupyter notebook
# esperamos que la transmisión termine en jupyter notebook
ssc.awaitTermination()

#-----Twitter Project-----#
"""
Primero necesitamos crear una cuenta de desarrollador de Twitter para obtener nuestros códigos de acceso.
Luego, deberá instalar la biblioteca tweepy, así como matplotlib y seaborn.
Comencemos yendo a: apps.twitter.com y creamos nueva app
Name: sparktwitternewapp
Description:: My spark app
website: https://www.google.com
aceptar developer agreement
Keys and access tokens , anotar de forma privada el consumer key (API key) y consumer secret (API secret)
Your access token, create my access token y anotar de forma privada el access token y access token secret
Instalamos las bibliotecas necesarias desde el terminal: pip3 install tweepy, matplotlib,seaborn
pip3 install pandas
"""

# Puede causar advertencias de desaprobación, seguro ignorar, no son errores
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import desc

# Solo podemos ejecutar esto una vez. Reiniciamos kernel por cualquier error.
sc = SparkContext()

ssc = StreamingContext(sc, 10 )
sqlContext = SQLContext(sc)
socket_stream = ssc.socketTextStream("127.0.0.1", 5555)
lines = socket_stream.window( 20 )

from collections import namedtuple
fields = ("tag", "count" )
Tweet = namedtuple( 'Tweet', fields )
# Usamos paréntesis para varias líneas o \.
( lines.flatMap( lambda text: text.split( " " ) ) # Se divide en una lista
  .filter( lambda word: word.lower().startswith("#") ) # Comprueba las llamadas de hashtag
  .map( lambda word: ( word.lower(), 1 ) ) # palabra en minúsculas
  .reduceByKey( lambda a, b: a + b ) # reducir por clave
  .map( lambda rec: Tweet( rec[0], rec[1] ) ) # Almacena en un objeto Tweet
  .foreachRDD( lambda rdd: rdd.toDF().sort( desc("count") ) # Los ordena en un DF
  .limit(10).registerTempTable("tweets") ) ) # Registramos en una tabla

# Desde la terminal ejecutamos: python TweetRead.py

ssc.start()    
# Veremos en la terminal los tweets

# A continuación seguimos ejecutando desde Jupyter notebook "Introduction to Spark Streaming.ipynb" para  ver gráficas de transmisiones en vivo:
import time
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns
# Only works for Jupyter Notebooks!
%matplotlib inline 

count = 0
while count < 10:
    
    time.sleep( 3 )
    top_10_tweets = sqlContext.sql( 'Select tag, count from tweets' )
    top_10_df = top_10_tweets.toPandas()
    display.clear_output(wait=True)
    sns.plt.figure( figsize = ( 10, 8 ) )
    sns.barplot( x="count", y="tag", data=top_10_df)
    sns.plt.show()
    count = count + 1


# Finalmente: 
ssc.stop()












