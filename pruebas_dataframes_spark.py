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
