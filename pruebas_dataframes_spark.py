# Spark dataFrame Basics
# pip install pyspark
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('Basics').getOrCreate()

df= spark.read.json('people.json')
df.show()
df.printSchema() # Esquema del dataframe
df.columns #solo nombre columnas 
df.describe() # Resumen estadístico del dataframe
df.describe().show() # Resumen estadístico de las columnas numéricas en su marco de datos

from pyspark.sql.types import (StructField,StringType,IntegerType,StructType) # Para aclarar el esquema y los tipos de columnas
                              
data_schema=[StructField('age',IntegerType(),True),
             StructField('name',StringType(),True)]
             
final_struc=StructType(fields=data_schema)
df= spark.read.json('people.json',schema=final_struc)
df.printSchema()
df['age']
type(df['age'])
df.select('age')
df.select('age').show()
type(df.select('age'))
df.head(2)
df.head(2)[0]
type(df.head(2)[0])
df.select(['age','name']) # Seleccionar varias columnas
df.select(['age','name']).show()
df.withColumn('newage',df['age']) # Creo nueva columna con los valores de age
df.withColumn('newage',df['age']).show()
df.withColumn('double_age',df['age']*2).show() # Creo nueva columna con los valores de age multiplicados por 2
df.show() # los cambios no son permanentes en el dataframe, hay que guardar lo anterior en una variable
df.withColumnRenamed('age','my_new_Age').show()
df.createOrReplaceTempView('people')
results= spark.sql("SELECT * FROM people")
results.show()
new_results= spark.sql("SELECT*FROM people WHERE age=30")
new_results.show()

# Spark dataFrame Basics Operations
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('ops').getOrCreate()

df=spark.read.csv('appl_stock.csv',inferSchema=True,header=True)
df.printSchema()
df.show()
df.head(3)
df.head(3)[0]
df.filter("Close <500").show()
df.filter("Close <500").select('Open').show()
df.filter("Close <500").select(['Open','Close']).show()
df.filter(df['Close']<500).show() # otra forma
df.filter(df['Close']<500).select('Volume').show()
df.filter ((df['Close'] <200) & (df['Open']>200)).show()
df.filter ((df['Close'] <200) & ~(df['Open']>200)).show()

df.filter(df['Low']== 197.16).show() # Otro Ejemplo
result=df.filter(df['Low']== 197.16).collect()
result
result[0]
row=result[0]
row.asDict()
row.asDict()['Volume']

# GroupBy and Aggregate Operations
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('aggs').getOrCreate()

df= spark.read.csv('sales_info.csv',inferSchema=True,header=True)
df.show()
df.printSchema()
df.groupBy("Company")
df.groupBy("Company").mean() 
df.groupBy("Company").mean().show() # mean() calcula el promedio de las empresas
df.groupBy("Company").sum().show() # sum() calcula la suma de los valores de cada empresa
df.groupBy("Company").max().show() # max() calcula el valor máximo de cada empresa
df.groupBy("Company").count().show() #count() cuenta las veces que aparece cada empresa
df.agg({'Sales':'sum'}).show() # Suma de todas las ventas
df.agg({'Sales':'max'}).show() #valor máximo de ventas
group_data= df.groupBy("company")
group_data.agg({'Sales':'max'}).show()

from pyspark.sql.functions import countDistinct,avg,stddev
df.select(countDistinct('Sales')).show()
df.select(avg('Sales')).show()
df.select(avg('Sales').alias('Average Sales')).show() # renombramos la función avg como Average Sales
df.select(stddev('Sales')).show()

from pyspark.sql.functions import format_number
sales_std=df.select(stddev("Sales").alias('std'))
sales_std.select(format_number('std',2)).show()
sales_std.select(format_number('std',2).alias('std')).show()
df.show()
df.orderBy("Sales").show() # Ordenamos el dataframe por número de ventas de menor a mayor
df.orderBy(df['Sales'].desc()).show() # Ordenamos el dataframe por número de ventas de mayor a menor

# Missing Data
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('miss').getOrCreate()

df= spark.read.csv('ContainsNull.csv',header=True,inferSchema=True)
df.show()
df.na.drop().show() # Imprime aquella fila donde no hay ningún null
df.na.drop(thresh=2).show() # Imprime aquella fila donde: thresh=1: todas filas (todas tienen 1 no null),thresh=2: filas con menos de 2 null o con 2 valores no null, thresh=3: filas sin  null o con 3 no null
df.na.drop(how='any').show() # Descarta las filas en las que hay algún valor null
df.na.drop(how='all').show() # Descarta las filas en las que todos los valores son null, en este caso ninguna
df.na.drop(subset='Sales').show() # Muestra solo filas donde se muestran valores de sales
df.printSchema()
df.na.fill('FILL VALUE').show() # Rellena cualquier valor string que sea null con FILL VALUE
df.na.fill(0).show() # Rellena con ceros los valores numéricos que sean null
df.na.fill('No name', subset=['Name']).show() # Rellena con No name los valores string null

from pyspark.sql.functions import mean
mean_val=df.select(mean(df['Sales'])).collect()
mean_val[0][0]
mean_sales=mean_val[0][0]
df.na.fill(mean_sales,['Sales']).show() # Rellena los valores null de Sales con el valor medio
df.na.fill(df.select(mean(df['Sales'])).collect()[0][0],['Sales']).show() # Lo mismo de antes en una sola línea # menos estético, menos recomendable 

# Dates and Timestamps
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('dates').getOrCreate()

df= spark.read.csv('appl_stock.csv',header=True,inferSchema=True)
df.head(1)
df.show()
df.select(['Date','Open']).show() # formato date: year-month-day hour

from pyspark.sql.functions import (dayofmonth,hour,
                                  dayofyear,month,
                                  year,weekofyear,
                                  format_number,date_format)
                                  
df.select(dayofmonth(df['Date'])).show()
df.select(hour(df['Date'])).show()
df.select(month(df['Date'])).show()
df.select(year(df['Date'])).show()
df.withColumn("Year",year(df['Date'])).show() # Añade al final la columna Year
newdf=df.withColumn("Year",year(df['Date']))
newdf.groupBy("Year").mean().show() # Calcula un promedio de los valores de las columnas para cada año
newdf.groupBy("Year").mean().select(["Year","avg(Close)" ]).show()  # Calcula un promedio de los valores de las columnas para cada año mostrando solo las columnas Year y avg(Close)
result=newdf.groupBy("Year").mean().select(["Year","avg(Close)" ])
result.show()
result.withColumnRenamed("avg(Close)","Average Closing Price").show() # Cambiamos de nombre avg(Close) a Average Closing Price
new= result.withColumnRenamed("avg(Close)","Average Closing Price")
new.select(['Year',format_number('Average Closing Price',2)]).show() # La columna ahora se llama (Average Closing Price) sus valores tienen 2 decimales
new.select(['Year',format_number('Average Closing Price',2).alias("avg(Close)")]).show() # Cambiamos el nombre a avg(Close)


#--------#
# Project Exercise : Use the walmart_stock.csv file to Answer and complete the tasks below!
# Start a simple Spark Session
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('walmart').getOrCreate()
# Load the Walmart Stock CSV File, have Spark infer the data types.
df=spark.read.csv('walmart_stock.csv',inferSchema=True,header=True)
df.show()
# What are the column names?
# ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
df.columns
#What does the Schema look like?
df.printSchema()
#Print out the first 5 columns.
df.head(5)
# Si queremos las líneas separadas
for row in df.head(5):
    print(row)
    print('\n')
# Use describe() to learn about the DataFrame
df.describe().show()
# Bonus Question!
# There are too many decimal places for mean and stddev in the describe() dataframe. 
# Format the numbers to just show up to two decimal places. 
# Pay careful attention to the datatypes that .describe() returns, 
# http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.cast
# Uh oh Strings! 
df.describe().printSchema()
from pyspark.sql.functions import (format_number)
result = df.describe()
result.select(result['summary'],
              format_number(result['Open'].cast('float'),2).alias('Open'),
              format_number(result['High'].cast('float'),2).alias('High'),
              format_number(result['Low'].cast('float'),2).alias('Low'),
              format_number(result['Close'].cast('float'),2).alias('Close'),
              result['Volume'].cast('int').alias('Volume')
             ).show()
df.describe()
# Create a new dataframe with a column called HV Ratio that is the ratio of the High Price versus volume of stock traded for a day
df2 = df.withColumn("HV Ratio",df["High"]/df["Volume"]) #.show()
df2.show()
df2.select('HV Ratio').show()
# What day had the Peak High in Price
from pyspark.sql.functions import (dayofyear)
# Didn't need to really do this much indexing
# Could have just shown the entire row
df.orderBy(df['High'].desc()).show()
df.orderBy(df["High"].desc()).head(1)[0][0]
# What is the mean of the Close column?
df.agg({'Close':'mean'}).show() 
#también:
from pyspark.sql.functions import mean
df.select(mean("Close")).show()
# What is the max and min of the Volume column?
df.agg({'Volume':'max'}).show() 
df.agg({'Volume':'min'}).show() 
# También: 
from pyspark.sql.functions import max,min
df.select(max("Volume"),min("Volume")).show()
# How many days was the Close lower than 60 dollars?
df.filter("Close < 60").count()
# También:
df.filter (df['Close'] <60).count()
# También: 
from pyspark.sql.functions import count
result = df.filter(df['Close'] < 60)
result.select(count('Close')).show()
# What percentage of the time was the High greater than 80 dollars ?
# In other words, (Number of Days High>80)/(Total Days in the dataset)
# 9.14 percent of the time it was over 80
# Many ways to do this
(df.filter(df["High"]>80).count()*1.0/df.count())*100
# What is the Pearson correlation between High and Volume?
# http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameStatFunctions.corr
from pyspark.sql.functions import corr
df.select(corr("High","Volume")).show()
# What is the max High per year?
from pyspark.sql.functions import year
yeardf = df.withColumn("Year",year(df["Date"])) # Añado columna year, se obtiene de Date
max_df = yeardf.groupBy('Year').max() # Selecciona los valores máximos de cada columna para cada año , Groupby lo usamos cuando tenemos varios valores repetidos en una columna, en este caso, los años en la columna Year.
# 2015
max_df.select('Year','max(High)').show()
# What is the average Close for each Calendar Month?
# In other words, across all the years, what is the average Close price for Jan,Feb, Mar, etc... Your result will have a value for each of these months.
from pyspark.sql.functions import month
monthdf = df.withColumn("Month",month("Date")) # Añado columna Month, se obtiene de Date
monthcolumn = monthdf.select("Month","Close") # Selecciono sólo columna Month donde aparece todos los meses
monthcolumn.show()
monthavgs = monthcolumn.select("Month","Close").groupBy("Month").mean() # Agrupamos por meses la columna monthcolumn y calculamos su media
monthavgs.select("Month","avg(Close)").orderBy('Month').show() # Ordenamos de menor a mayor


