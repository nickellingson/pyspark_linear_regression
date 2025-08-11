from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark=SparkSession.builder.appName('ML').getOrCreate()

training = spark.read.csv('Book3.csv', header=True, inferSchema=True)
training.show()
print(training.printSchema())

# Create feature
featureassembler = VectorAssembler(inputCols=["Age", "Experience"], outputCol="Independent Features")
output = featureassembler.transform(training)
output.show()
finalized_data = output.select("Independent Features", "Salary")
finalized_data.show()

# Train test split
train_data, test_data = finalized_data.randomSplit([.75,.25])
regressor = LinearRegression(featuresCol="Independent Features", labelCol="Salary")
regressor = regressor.fit(train_data)
print(regressor.coefficients)
print(regressor.intercept)

# Prediction
pred_results=regressor.evaluate(test_data)
pred_results.predictions.show()

print(pred_results.meanAbsoluteError, pred_results.meanSquaredError)
