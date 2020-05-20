""""
Este es el campo del aprendizaje automático que se centra en la creación de modelos a partir de una fuente de datos de texto (directamente de artículos de palabras).

Sugerencias opcionales de lectura:
Artículo de Wikipedia sobre NLP
Libro NLTK (biblioteca de Python separada)
Fundamentos del procesamiento estadístico del lenguaje natural (Manning)

Ejemplos de NLP:
-Clustering News Articles
-Suggesting similar books
-Grouping Legal Documents
-Analyzing Consumer Feedback
-Spam Email Detection

Nuestro proceso básico para NLP:
Compilar todos los documentos (Corpus)
Caracterizar las palabras a números
Comparar características de documentos


Una forma estándar de hacerlo es mediante el uso de lo que se conoce como métodos "TF-IDF".
TF-IDF significa Frecuencia de término - Frecuencia de documento inversa

Ejemplo simple:
Tienes 2 documentos:
"Casa Azul"
"Casa Roja"
Caraceterización basado en el conteo de palabras:
"Casa Azul" -> (rojo, azul, casa) -> (0,1,1)
"Casa Roja" -> (rojo, azul, casa) -> (1,0,1)


Un documento representado como un vector de recuento de palabras se denomina "Bolsa de palabras"
"Casa Azul" -> (rojo, azul, casa) -> (0,1,1)
"Casa Roja" -> (rojo, azul, casa) -> (1,0,1)
Estos son ahora vectores en un espacio N-dimensional, podemos comparar vectores con similitud de coseno.

Podemos mejorar la Bolsa de palabras ajustando el recuento de palabras según su frecuencia en el corpus (el grupo de todos los documentos)
Podemos usar TF-IDF.

Término Frecuencia: Importancia del término dentro de ese documento.
TF (x, y) = Número de apariciones del término x en el documento y

Frecuencia de documento inversa: importancia del término en el corpus.
IDF (t) = log (N / dfx) donde
N = número total de documentos
dfx = número de documentos con el término.
"""

# TOOLS FOR NLP PART ONE
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('nlp').getOrCreate()
from pyspark.ml.feature import Tokenizer,RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
# sentence dataframe
sen_df= spark.createDataFrame([
    (0,'Hi I heard about Spark'),
    (1,'I wish java could use case classes'),
    (2,'Logistic,regression,models,are,neat')
],['id','sentence'])

sen_df.show() 
"""
+---+--------------------+
| id|            sentence|
+---+--------------------+
|  0|Hi I heard about ...|
|  1|I wish java could...|
|  2|Logistic,regressi...|
+---+--------------------+
"""
tokenizer= Tokenizer(inputCol='sentence',outputCol='words')
regex_tokenizer=RegexTokenizer(inputCol='sentence',outputCol='words',
                              pattern='\\W')

count_tokens=udf(lambda words:len (words),IntegerType())
tokenized=tokenizer.transform(sen_df)
tokenized.show()
"""
+---+--------------------+--------------------+
| id|            sentence|               words|
+---+--------------------+--------------------+
|  0|Hi I heard about ...|[hi, i, heard, ab...|
|  1|I wish java could...|[i, wish, java, c...|
|  2|Logistic,regressi...|[logistic,regress...|
+---+--------------------+--------------------+
"""
# Si observamos la tercera oración, se trata como una cadena gigante, de 1 solo token, al no haber espacio, sino comas.
tokenized.withColumn('tokens',count_tokens(col('words'))).show()
"""
+---+--------------------+--------------------+------+
| id|            sentence|               words|tokens|
+---+--------------------+--------------------+------+
|  0|Hi I heard about ...|[hi, i, heard, ab...|     5|
|  1|I wish java could...|[i, wish, java, c...|     7|
|  2|Logistic,regressi...|[logistic,regress...|     1|
+---+--------------------+--------------------+------+
"""
rg_tokenized= regex_tokenizer.transform(sen_df)
# Ahora en la última oración tenemos 5 tokens
rg_tokenized.withColumn('tokens',count_tokens(col('words'))).show()
"""
+---+--------------------+--------------------+------+
| id|            sentence|               words|tokens|
+---+--------------------+--------------------+------+
|  0|Hi I heard about ...|[hi, i, heard, ab...|     5|
|  1|I wish java could...|[i, wish, java, c...|     7|
|  2|Logistic,regressi...|[logistic, regres...|     5|
+---+--------------------+--------------------+------+
"""
from pyspark.ml.feature import StopWordsRemover
sentenceDataFrame= spark.createDataFrame([
    (0,['I','saw','the','green','horse']),
    (1,['Mary','had','a','little','lamb'])
],['id','tokens'])
sentenceDataFrame.show()
"""
+---+--------------------+
| id|              tokens|
+---+--------------------+
|  0|[I, saw, the, gre...|
|  1|[Mary, had, a, li...|
+---+--------------------+
"""
remover= StopWordsRemover(inputCol='tokens',outputCol='filtered')
remover.transform(sentenceDataFrame).show()
"""
+---+--------------------+--------------------+
| id|              tokens|            filtered|
+---+--------------------+--------------------+
|  0|[I, saw, the, gre...| [saw, green, horse]|
|  1|[Mary, had, a, li...|[Mary, little, lamb]|
+---+--------------------+--------------------+
"""
# n-gram
from pyspark.ml.feature import NGram
wordDataFrame= spark.createDataFrame([
    (0,["Hi", "I", "heard", "about", "Spark"]),
    (1,["I", "wish", "java", "could", "use", "case", "classes"]),
    (2,["Logistic", "regression", "models", "are", "neat"]),
],['id','words'])

ngram= NGram(n=2, inputCol='words',outputCol='grams')
ngram.transform(wordDataFrame).show()
"""
+---+--------------------+--------------------+
| id|               words|               grams|
+---+--------------------+--------------------+
|  0|[Hi, I, heard, ab...|[Hi I, I heard, h...|
|  1|[I, wish, java, c...|[I wish, wish jav...|
|  2|[Logistic, regres...|[Logistic regress...|
+---+--------------------+--------------------+
"""
ngram.transform(wordDataFrame).select('grams').show()
"""
+--------------------+
|               grams|
+--------------------+
|[Hi I, I heard, h...|
|[I wish, wish jav...|
|[Logistic regress...|
+--------------------+
"""
ngram.transform(wordDataFrame).select('grams').show(truncate=False)
"""
+------------------------------------------------------------------+
|grams                                                             |
+------------------------------------------------------------------+
|[Hi I, I heard, heard about, about Spark]                         |
|[I wish, wish java, java could, could use, use case, case classes]|
|[Logistic regression, regression models, models are, are neat]    |
+------------------------------------------------------------------+
"""
ngram= NGram(n=3, inputCol='words',outputCol='grams')
ngram.transform(wordDataFrame).select('grams').show(truncate=False)
"""
+--------------------------------------------------------------------------------+
|grams                                                                           |
+--------------------------------------------------------------------------------+
|[Hi I heard, I heard about, heard about Spark]                                  |
|[I wish java, wish java could, java could use, could use case, use case classes]|
|[Logistic regression models, regression models are, models are neat]            |
+----------
"""
# TOOLS FOR NLP PART TWO
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('nlp').getOrCreate()
from pyspark.ml.feature import HashingTF,IDF,Tokenizer
sentenceData= spark.createDataFrame([
    (0.0,'Hi I heard about Spark'),
    (0.0,'I wish java could use case classes'),
    (1.0,'Logistic regression models are neat')
],['label','sentence'])
sentenceData.show()
"""
+-----+--------------------+
|label|            sentence|
+-----+--------------------+
|  0.0|Hi I heard about ...|
|  0.0|I wish java could...|
|  1.0|Logistic regressi...|
+-----+--------------------+
"""
tokenizer= Tokenizer(inputCol='sentence',outputCol='words')
words_data=tokenizer.transform(sentenceData)
words_data.show()
"""
+-----+--------------------+--------------------+
|label|            sentence|               words|
+-----+--------------------+--------------------+
|  0.0|Hi I heard about ...|[hi, i, heard, ab...|
|  0.0|I wish java could...|[i, wish, java, c...|
|  1.0|Logistic regressi...|[logistic, regres...|
+-----+--------------------+--------------------+
"""
words_data.show(truncate=False)
"""
+-----+-----------------------------------+------------------------------------------+
|label|sentence                           |words                                     |
+-----+-----------------------------------+------------------------------------------+
|0.0  |Hi I heard about Spark             |[hi, i, heard, about, spark]              |
|0.0  |I wish java could use case classes |[i, wish, java, could, use, case, classes]|
|1.0  |Logistic regression models are neat|[logistic, regression, models, are, neat] |
+-----+-----------------------------------+------------------------------------------+
"""
hasing_tf=HashingTF(inputCol='words',outputCol='rawFeatures')
featurized_data= hasing_tf.transform(words_data)
idf=IDF(inputCol='rawFeatures',outputCol='features')
idf_model=idf.fit(featurized_data)
rescaled_data=idf_model.transform(featurized_data)
rescaled_data.select('label','features').show()
"""
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  0.0|(262144,[24417,49...|
|  0.0|(262144,[20719,24...|
|  1.0|(262144,[13671,91...|
+-----+--------------------+
"""
rescaled_data.select('label','features').show(truncate=False)
"""
+-----+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|label|features                                                                                                                                                                                        |
+-----+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|0.0  |(262144,[24417,49304,73197,91137,234657],[0.28768207245178085,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])                                                     |
|0.0  |(262144,[20719,24417,55551,116873,147765,162369,192310],[0.6931471805599453,0.28768207245178085,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])|
|1.0  |(262144,[13671,91006,132713,167122,190884],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])                                                    |
+-----+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
"""

from pyspark.ml.feature import CountVectorizer
df= spark.createDataFrame([
    (0,"a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])

df.show()
"""
+---+---------------+
| id|          words|
+---+---------------+
|  0|      [a, b, c]|
|  1|[a, b, b, c, a]|
+---+---------------+
"""

cv= CountVectorizer(inputCol='words',outputCol='features',
                   vocabSize=3,minDF=2.0)

model= cv.fit(df)
result=model.transform(df)
# Método de la Bolsa de palabras
result.show(truncate=False)
"""
+---+---------------+-------------------------+
|id |words          |features                 |
+---+---------------+-------------------------+
|0  |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
|1  |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
+---+---------------+-------------------------+
"""

#-----NLP Code Along: Filtro de detección de spam-----#
# Nuestro conjunto de datos consta de mensajes de texto voluntarios de un estudio en Singapur y algunos mensajes de spam de un sitio de informes del Reino Unido.

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('nlp').getOrCreate()
data= spark.read.csv('smsspamcollection/SMSSpamCollection',inferSchema=True,sep='\t')
data.show()
"""
+----+--------------------+
| _c0|                 _c1|
+----+--------------------+
| ham|Go until jurong p...|
| ham|Ok lar... Joking ...|
|spam|Free entry in 2 a...|
| ham|U dun say so earl...|
| ham|Nah I don't think...|
|spam|FreeMsg Hey there...|
| ham|Even my brother i...|
| ham|As per your reque...|
|spam|WINNER!! As a val...|
|spam|Had your mobile 1...|
| ham|I'm gonna be home...|
|spam|SIX chances to wi...|
|spam|URGENT! You have ...|
| ham|I've been searchi...|
| ham|I HAVE A DATE ON ...|
|spam|XXXMobileMovieClu...|
| ham|Oh k...i'm watchi...|
| ham|Eh u remember how...|
| ham|Fine if thats th...|
|spam|England v Macedon...|
+----+--------------------+
only showing top 20 rows
"""

data= data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data.show()
"""
+-----+--------------------+
|class|                text|
+-----+--------------------+
|  ham|Go until jurong p...|
|  ham|Ok lar... Joking ...|
| spam|Free entry in 2 a...|
|  ham|U dun say so earl...|
|  ham|Nah I don't think...|
| spam|FreeMsg Hey there...|
|  ham|Even my brother i...|
|  ham|As per your reque...|
| spam|WINNER!! As a val...|
| spam|Had your mobile 1...|
|  ham|I'm gonna be home...|
| spam|SIX chances to wi...|
| spam|URGENT! You have ...|
|  ham|I've been searchi...|
|  ham|I HAVE A DATE ON ...|
| spam|XXXMobileMovieClu...|
|  ham|Oh k...i'm watchi...|
|  ham|Eh u remember how...|
|  ham|Fine if thats th...|
| spam|England v Macedon...|
+-----+--------------------+
only showing top 20 rows
"""

from pyspark.sql.functions import length
data= data.withColumn('length',length(data['text']))
data.show()
"""
+-----+--------------------+------+
|class|                text|length|
+-----+--------------------+------+
|  ham|Go until jurong p...|   111|
|  ham|Ok lar... Joking ...|    29|
| spam|Free entry in 2 a...|   155|
|  ham|U dun say so earl...|    49|
|  ham|Nah I don't think...|    61|
| spam|FreeMsg Hey there...|   147|
|  ham|Even my brother i...|    77|
|  ham|As per your reque...|   160|
| spam|WINNER!! As a val...|   157|
| spam|Had your mobile 1...|   154|
|  ham|I'm gonna be home...|   109|
| spam|SIX chances to wi...|   136|
| spam|URGENT! You have ...|   155|
|  ham|I've been searchi...|   196|
|  ham|I HAVE A DATE ON ...|    35|
| spam|XXXMobileMovieClu...|   149|
|  ham|Oh k...i'm watchi...|    26|
|  ham|Eh u remember how...|    81|
|  ham|Fine if thats th...|    56|
| spam|England v Macedon...|   155|
+-----+--------------------+------+
only showing top 20 rows
"""
data.groupBy('class').mean().show()
"""
+-----+-----------------+
|class|      avg(length)|
+-----+-----------------+
|  ham|71.45431945307645|
| spam|138.6706827309237|
+-----+-----------------+
"""

from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer,
                               IDF, StringIndexer)
tokenizer= Tokenizer(inputCol='text',outputCol='token_text')
stop_remove=StopWordsRemover(inputCol='token_text',outputCol='stop_token')
count_vec= CountVectorizer(inputCol='stop_token',outputCol='c_vec')
idf=IDF(inputCol='c_vec',outputCol='tf_idf')
ham_spam_to_numeric= StringIndexer(inputCol='class',outputCol='label')

from pyspark.ml.feature import VectorAssembler
clean_up= VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

from pyspark.ml.classification import NaiveBayes
nb= NaiveBayes()

from pyspark.ml import Pipeline
data_prep_pipe= Pipeline(stages=[ham_spam_to_numeric,tokenizer,
                                stop_remove,count_vec,idf,clean_up])
cleaner=data_prep_pipe.fit(data)
clean_data=cleaner.transform(data)
clean_data.columns
"""
['class',
 'text',
 'length',
 'label',
 'token_text',
 'stop_token',
 'c_vec',
 'tf_idf',
 'features']
"""

clean_data=clean_data.select('label','features')
clean_data.show()
"""
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  0.0|(13424,[7,11,31,6...|
|  0.0|(13424,[0,24,297,...|
|  1.0|(13424,[2,13,19,3...|
|  0.0|(13424,[0,70,80,1...|
|  0.0|(13424,[36,134,31...|
|  1.0|(13424,[10,60,139...|
|  0.0|(13424,[10,53,103...|
|  0.0|(13424,[125,184,4...|
|  1.0|(13424,[1,47,118,...|
|  1.0|(13424,[0,1,13,27...|
|  0.0|(13424,[18,43,120...|
|  1.0|(13424,[8,17,37,8...|
|  1.0|(13424,[13,30,47,...|
|  0.0|(13424,[39,96,217...|
|  0.0|(13424,[552,1697,...|
|  1.0|(13424,[30,109,11...|
|  0.0|(13424,[82,214,47...|
|  0.0|(13424,[0,2,49,13...|
|  0.0|(13424,[0,74,105,...|
|  1.0|(13424,[4,30,33,5...|
+-----+--------------------+
only showing top 20 rows
"""
training,test=clean_data.randomSplit([0.7,0.3])
spam_detector=nb.fit(training)
data.printSchema()
"""
root
 |-- class: string (nullable = true)
 |-- text: string (nullable = true)
 |-- length: integer (nullable = true)
"""
test_results= spam_detector.transform(test)
test_results.show()
"""
+-----+--------------------+--------------------+--------------------+----------+
|label|            features|       rawPrediction|         probability|prediction|
+-----+--------------------+--------------------+--------------------+----------+
|  0.0|(13424,[0,1,4,50,...|[-830.19229371859...|[1.0,7.1865697329...|       0.0|
|  0.0|(13424,[0,1,7,15,...|[-656.19608717291...|[1.0,3.7522464876...|       0.0|
|  0.0|(13424,[0,1,9,14,...|[-539.18915845895...|[1.0,9.5065245384...|       0.0|
|  0.0|(13424,[0,1,12,33...|[-454.53625070496...|[1.0,1.2828752326...|       0.0|
|  0.0|(13424,[0,1,14,18...|[-1366.2835003483...|[1.0,5.9348099662...|       0.0|
|  0.0|(13424,[0,1,15,20...|[-684.53383879069...|[1.0,5.1261818260...|       0.0|
|  0.0|(13424,[0,1,20,27...|[-967.78516925213...|[1.0,1.2918828208...|       0.0|
|  0.0|(13424,[0,1,21,27...|[-760.33177790758...|[1.0,1.7839552118...|       0.0|
|  0.0|(13424,[0,1,23,63...|[-1294.6486382999...|[1.0,3.1061933096...|       0.0|
|  0.0|(13424,[0,1,24,31...|[-355.73202891079...|[1.0,9.0190240887...|       0.0|
|  0.0|(13424,[0,1,27,35...|[-1485.9667899645...|[0.99999997455451...|       0.0|
|  0.0|(13424,[0,1,43,69...|[-616.31896474537...|[0.99998808877209...|       0.0|
|  0.0|(13424,[0,1,416,6...|[-300.05701948277...|[0.99999999999999...|       0.0|
|  0.0|(13424,[0,2,3,5,6...|[-2572.2209931774...|[1.0,2.0649701650...|       0.0|
|  0.0|(13424,[0,2,3,6,9...|[-3295.7663122378...|[1.0,9.63149943E-...|       0.0|
|  0.0|(13424,[0,2,4,5,7...|[-995.44841272981...|[1.0,3.3751903107...|       0.0|
|  0.0|(13424,[0,2,4,7,2...|[-512.67288900125...|[1.0,4.1013386632...|       0.0|
|  0.0|(13424,[0,2,4,8,2...|[-568.69782770176...|[1.0,2.4576072052...|       0.0|
|  0.0|(13424,[0,2,4,10,...|[-1219.4411622787...|[1.0,9.8986542758...|       0.0|
|  0.0|(13424,[0,2,4,40,...|[-1580.3026844172...|[0.99999999999993...|       0.0|
+-----+--------------------+--------------------+--------------------+----------+
only showing top 20 rows
"""
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval=MulticlassClassificationEvaluator()
acc=acc_eval.evaluate(test_results)
print('ACC of NB Model')
print(acc) # Obtengo 0.9296950535275138 que es un 92 % de precisión

# Ahora podemos probar a cambiar por otro modelo : from pyspark.ml.classification import NaiveBayes
# Y ver si la precisión del modelo mejora














