""""
Este es el campo del aprendizaje automático que se centra en la creación de modelos a partir de una fuente de datos de texto (directamente de artículos de palabras).

Sugerencias opcionales de lectura:
Artículo de Wikipedia sobre PNL
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

#-----NLP Code Along-----#















