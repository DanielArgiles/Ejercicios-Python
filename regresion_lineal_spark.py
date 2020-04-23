# http://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('lrex').getOrCreate()
from pyspark.ml.regression import LinearRegression
# Load training data
training = spark.read.format("libsvm").load("sample_linear_regression_data.txt")
# This is the format that Spark expects. Two columns with the names "label" and "features".
# The "label" column then needs to have the numerical label, either a regression numerical value, or a numerical value that matches to a classification grouping. Later on we will talk about unsupervised learning algorithms that by their nature do not use or require a label.
# The feature column has inside of it a vector of all the features that belong to that row. Usually what we end up doing is combining the various feature columns we have into a single 'features' column using the data transformations we've learned about.
training.show()
# These are the default values for the featuresCol, labelCol, predictionCol
# You could also pass in additional parameters for regularization, do the reading 
# in ISLR to fully understand that, after that its just some simple parameter calls.
# Check the documentation with Shift+Tab for more info!
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
# Fit the model
lrModel=lr.fit(training)
# Print the coefficients and intercept for linear regression
lrModel.coefficients
lrModel.intercept
# Summarize the model over the training set and print out some metrics
training_summary=lrModel.summary
training_summary.r2
training_summary.rootMeanSquaredError
#Train/Test Splits
all_data=spark.read.format("libsvm").load("sample_linear_regression_data.txt")
# Pass in the split between training/test as a list.
# No correct, but generally 70/30 or 60/40 splits are used. 
# Depending on how much data you have and how unbalanced it is.
train_data,test_data=all_data.randomSplit([0.7,0.3])
train_data,test_data # 2 dataframes : 70% train data, 30% test data
train_data.show()
train_data.describe().show()
test_data.describe().show()
correct_model=lr.fit(train_data)
test_results=correct_model.evaluate(test_data)
test_results.residuals.show()
test_results.rootMeanSquaredError
unlabeled_data=test_data.select('features')
unlabeled_data.show()
predictions=correct_model.transform(unlabeled_data)
predictions.show()

#-----#
# Linear Regression Custom Example
# We’ll examine some Ecommerce Customer Data for a company's website and mobile app
# The main idea will be trying to predict a customer’s total amount expenditure (a continuous money value).
# We’ll also discuss how to convert realistic data into a format accepted by Spark’s MLlib!
# Then we want to see if we can build a regression model that will predict the customer's yearly spend on the company's product.
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName('lr_example').getOrCreate()
from pyspark.ml.regression import LinearRegression
data=spark.read.csv("Ecommerce_Customers.csv",inferSchema=True,header=True)
data.printSchema()
data.show()
# 1st customer
for item in data.head(1)[0]:
    print(item)
# 2ns customer
for item in data.head(2)[1]:
    print(item)
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
data.columns
assembler=VectorAssembler(inputCols=['Avg Session Length','Time on App','Time on Website','Length of Membership']
                          ,outputCol='features')
output=assembler.transform(data)
output.printSchema()
output.select('features').show()
output.head(1)
final_data=output.select('features','Yearly Amount Spent')
final_data.show()
train_data,test_data= final_data.randomSplit([0.7,0.3])
train_data.describe().show()
test_data.describe().show()
lr=LinearRegression(labelCol='Yearly Amount Spent')
lr_model= lr.fit(train_data)
test_results= lr_model.evaluate(test_data)
test_results.residuals.show()
test_results.rootMeanSquaredError
test_results.r2
final_data.describe().show()
unlabeled_data= test_data.select('features')
unlabeled_data.show()
predictions=lr_model.transform(unlabeled_data)
predictions.show()
