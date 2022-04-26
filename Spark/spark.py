# solutions.py

import pyspark
from pyspark.sql import SparkSession
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE
#import logist regression for testing purposes
from pyspark.ml.classification import LogisticRegression



# --------------------- Resilient Distributed Datasets --------------------- #

### Problem 1
def word_count(filename='huck_finn.txt'):
    """
    A function that counts the number of occurrences unique occurrences of each
    word. Sorts the words by count in descending order.
    Parameters:
        filename (str): filename or path to a text file
    Returns:
        word_counts (list): list of (word, count) pairs for the 20 most used words
    """
    #start spark session
    spark = SparkSession\
            .builder\
            .appName('app_name')\
            .getOrCreate()

    #read in textfile into spark RDD object
    huck = spark.sparkContext.textFile(filename)
    #flatten the text
    huck = huck.flatMap(lambda row: row.split())
    #get the count for each word
    huck = huck.map(lambda row: (row, 1))
    huck = huck.reduceByKey(lambda x, y: x+y)
    #sort by the count and get the first 20 elements of the sorted list
    counts = huck.sortBy(lambda row: row[1], ascending=False).collect()[:20]
    #stop spark
    spark.stop()
    #return counts
    return counts


### Problem 2
def monte_carlo(n=10**5, parts=6):
    """
    Runs a Monte Carlo simulation to estimate the value of pi.
    Parameters:
        n (int): number of sample points per partition
        parts (int): number of partitions
    Returns:
        pi_est (float): estimated value of pi
    """

    #start spark session
    spark = SparkSession\
            .builder\
            .appName('app_name')\
            .getOrCreate()

    #parallelize uniform draw
    uniform_draw = spark.sparkContext.parallelize(np.random.uniform(low=-1, high=1, size=( n*parts, 2)), parts)
    #get distances
    distances = uniform_draw.map(np.linalg.norm)
    #filter by point inside
    inside_region = distances.filter(lambda dist: dist <= 1)
    #get total count of points left inside region
    count_inside = inside_region.count()
    #stop spark
    spark.stop()
    #return estimate
    return 4 * count_inside / (n*parts)

# ------------------------------- DataFrames ------------------------------- #

### Problem 3
def titanic_df(filename='titanic.csv'):
    """
    Calculates some statistics from the titanic data.

    Returns: the number of women on-board, the number of men on-board,
             the survival rate of women,
             and the survival rate of men in that order.
    """
    #start spark session
    spark = SparkSession\
            .builder\
            .appName('app_name')\
            .getOrCreate()

    #create schema
    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT' )
    #create titanic dataframe
    titanic = spark.read.csv(filename, schema=schema)
    #get dataframe of men
    men = titanic.filter(titanic.sex == 'male')
    #get dataframe of men that survived
    men_survived = men.filter(men.survived==1)
    #get dataframe of women
    women = titanic.filter(titanic.sex == 'female')
    #get dataframe of women that survived
    women_survived = women.filter(women.survived==1)

    #count everyhting up
    num_men = int(men.count())
    num_men_survived = int(men_survived.count())
    num_women = women.count()
    num_women_survived = women_survived.count()
    #stop spark
    spark.stop()
    #return the correct count and rates
    return num_women,  num_men, num_women_survived/num_women, num_men_survived/num_men

### Problem 4
def crime_and_income(crimefile='london_crime_by_lsoa.csv',
                     incomefile='london_income_by_borough.csv', major_cat='Robbery'):
    """
    Explores crime by borough and income for the specified min_cat
    Parameters:
        crimefile (str): path to csv file containing crime dataset
        incomefile (str): path to csv file containing income dataset
        major_cat (str): crime minor category to analyze
    returns:
        numpy array: borough names sorted by percent months with crime, descending
    """

    spark = SparkSession\
            .builder\
            .appName('app_name')\
            .getOrCreate()

    #read in the the crime and income dataframes
    crime_df = spark.read.csv(crimefile, header=True, inferSchema=True)
    income_df = spark.read.csv(incomefile, header=True, inferSchema=True)
    #select only the columns we care about from the crime dataframe
    new_df = crime_df.select(['borough', 'major_category', 'value'])
    #filter by category, then group by borough and sum by the value
    new_df = new_df.filter(new_df.major_category == major_cat).groupBy('borough').sum('value')
    #join on borough
    new_df = new_df.join(income_df, on='borough').select(['borough', 'sum(value)', 'median-08-16'])
    #get the final data by sorting by the average crime value
    final_arr = np.array(new_df.sort('sum(value)', ascending=False).collect())
    #stop spark
    spark.stop()
    #scatter plot
    plt.scatter(final_arr[:, 1].astype(np.float64), final_arr[:, 2].astype(np.float64))
    plt.xlabel('Crime Number')
    plt.ylabel('Meidan Income')
    plt.show()
    #return final data
    return final_arr


### Problem 5
def titanic_classifier(filename='titanic.csv'):
    """
    Implements a classifier model to predict who survived the Titanic.
    Parameters:
        filename (str): path to the dataset
    Returns:
        metrics (list): a list of metrics gauging the performance of the model
            ('accuracy', 'weightedPrecision', 'weightedRecall')
    """
    spark = SparkSession\
            .builder\
            .appName('app_name')\
            .getOrCreate()

    #schema for titantic dataset
    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT' )
    #create titanic dataframe
    titanic = spark.read.csv(filename, schema=schema)

    #convert sex column to binary categorical variable
    sex_binary = StringIndexer(inputCol='sex', outputCol='sex_binary')

    #on hot encode pclass
    onehot = OneHotEncoder(inputCols=['pclass'], outputCols=['pclass_onehot'])

    #create single features column
    features = ['sex_binary', 'pclass_onehot', 'age', 'sibsp', 'parch', 'fare']
    features_col = VectorAssembler(inputCols=features, outputCol='features')

    #create a transform pipeline
    pipeline = Pipeline(stages=[sex_binary, onehot, features_col])
    titanic = pipeline.fit(titanic).transform(titanic)

    #drop unecessary columns
    titanic = titanic.drop('pclass', 'name', 'sex')

    #75/25 train-test-split
    train, test = titanic.randomSplit([0.75, 0.25], seed=11)

    #initialize our random forest classifier
    RF = RandomForestClassifier(labelCol = 'survived', featuresCol = 'features')

    #run a train-test-validation split on best elastic net params
    paramGrid = ParamGridBuilder()\
                .addGrid(RF.numTrees, np.arange(20, 40))\
		        .addGrid(RF.maxDepth, np.arange(10, 15))\
		        .build()

    #will try all combinations and get the best one
    rf_tvs = TrainValidationSplit(estimator=RF, 
                                    estimatorParamMaps=paramGrid,
                                    evaluator=MCE(labelCol='survived'),
                                    trainRatio=0.75,
                                    seed=11)

    #train the classifier by fitting our tvs object to the training set
    rf_clf = rf_tvs.fit(train)
    #use best fit model to evaluate the test data

    results = rf_clf.bestModel.evaluate(test)
    results.predictions.select(['survived', 'prediction'])
    # .show(5)
    #get the performance
    accuracy, weighted_recall, weighted_precision = results.accuracy, results.weightedRecall, results.weightedPrecision

    spark.stop()

    return accuracy, weighted_recall, weighted_precision

if __name__ == "__main__":
    # print(word_count())
    # print(monte_carlo())
    # print(titanic_df())
    # print(crime_and_income())
    # print(titanic_classifier())
    pass
