# Pyspark
###### tags: `Python` `Pyspark` `Spark`
[TOC]

[有目錄版HackMD筆記](https://hackmd.io/yUxnYOU9TL2ocN6z0CCS_w?both)

**Resource :**[Introduction to PySpark](https://www.datacamp.com/courses/introduction-to-pyspark)

Data structure
-
Spark's core data structure is the Resilient Distributed Dataset (RDD)->Spark DataFrame abstraction built on top of RDDs.

**Method to create data interface**
```python
# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)
```

**Method to returns the names of all the tables in your cluster as a list.**
`print(spark.catalog.listTables())`

**Running a query on this table:sql()**
```python
# Don't change this query
query = "FROM flights SELECT * LIMIT 10"
# Get the first 10 rows of flights
#we've already created a SparkSession called spark
flights10 = spark.sql(query)
# Show the results
flights10.show()
```

**transform spark dataframe to panda dataframe:.toPandas()**
```python
# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"
# Run the query
flight_counts =spark.sql(query)
# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()
# Print the head of pd_counts
print(pd_counts.head())
```

**Transform panda dataframe to spark dataframe:.createDataFrame()**
```python
# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))
# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)
```

**Registers the DataFrame as a table in the catalog:.createTempView()**
- This method registers the DataFrame as a table in the catalog, but as this table is temporary, it can only be accessed from the specific SparkSession used to create the Spark DataFrame.
- There is also the method .createOrReplaceTempView(). This safely creates a new temporary table if nothing was there before, or updates an existing table if one was already defined.
```python
# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")
# Examine the tables in the catalog again
print(spark.catalog.listTables())
```

**Create a DataFrame from a .csv file**
```python
# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"
# Read in the airports data
airports = spark.read.csv(file_path,header=True)
# Show the data
airports.show()
```

**Creating columns**
```python
# Create the DataFrame flights
flights = spark.table("flights")
# Show the head
print(flights.show())
# Add duration_hrs
flights = flights.withColumn("duration_hrs",flights.air_time/60)
```
![](https://i.imgur.com/KeVizav.png)

Machine Learning Pipelines
-
**pyspark.ml module**
- At the core of the pyspark.ml module are the Transformer and Estimator classes.
    - Transformer classes have a .transform() method that takes a DataFrame and returns a new DataFrame; usually the original one with a new column appended.
    - Estimator classes all implement a .fit() method. These methods also take a DataFrame, but instead of returning another DataFrame they return a model object. This can be something like a StringIndexerModel for including categorical data saved as strings in your models, or a RandomForestModel that uses the random forest algorithm for classification or regression.
    

**Join the dataframe**
```python
# Rename year column
planes=planes.withColumnRenamed("year","plane_year")
# Join the DataFrames (flights+plane, keycolumn=tailonum)
model_data = flights.join(planes, on="tailnum", how="leftouter")
```

**String to integer**
- Spark only handles numeric data (integer or double)
- .withColumn() +df.col.cast("new type")
```python
# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
```

**Making a Boolean**
```python
# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)
# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))
```

**pyspark.ml.features submodule**
- solve categorical to numeric feature:'one-hot vectors`
    - First: create a StringIndexer,  Members of this class are Estimators that take a DataFrame with a column of strings and map each unique string to a number
    - Second: encode this numeric column as a one-hot vector using a OneHotEncoder
    ```python
    # Create a StringIndexer
    dest_indexer = StringIndexer(inputCol="dest",outputCol="dest_index")
    # Create a OneHotEncoder
    dest_encoder =OneHotEncoder(inputCol="dest_index",outputCol="dest_fact")
    ```
- Combine and pick more columns
```python
# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")
```

**Create the pipeline**
- "stages" should be a list holding all the stages you want your data to go through in the pipeline
```python
# Import Pipeline
from pyspark.ml import Pipeline
# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
```

**Train, Test**
```python
# Split the data into training 60% and test sets 40% 
training, test = piped_data.randomSplit([.6, .4])
```

**Estimator:Logistic regression**
- Create model
```python
# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression
# Create a LogisticRegression Estimator
lr =LogisticRegression()
```

**Create the evaluator**
```python
# Import the evaluation submodule
import pyspark.ml.evaluation as evals
# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
```

**Make a grid**
- create a grid of values to search over when looking for the optimal hyperparameters. 
    - The .addGrid() method takes a model parameter
    - The .build() method takes no arguments, it just returns the grid that you'll use later.
```python
# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()
# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
# Build the grid
grid = grid.build()
```

**Make the validator**

```python
# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator)
```

**Fit the model**
```Python
# Fit cross validation models
models = cv.fit(training)
# Extract the best model
best_lr = models.bestModel
```
- without crossvalidation(parameter values regParam=0 and elasticNetParam=0 as default)
```python
# Call lr.fit()
best_lr =lr.fit(training)
# Print best_lr
print(best_lr)
```

**Evaluate the model**
```python
# Use the model to predict the test set
test_results = best_lr.transform(test)
# Evaluate the predictions (AUC)
print(evaluator.evaluate(test_results))
```
