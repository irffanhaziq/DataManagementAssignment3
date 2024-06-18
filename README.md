# Iris : Classification Using Spark MLlib

## Project Summary

This project involves the classification tasks on the Iris dataset using Spark MLlib. The key steps covered in this project include:

1.  Loading packages needed
2.  Create a SparkSession
3.  Load the iris dataset
4.  Selecting a classifier, specifically the Random Forest classifier.
5.  Fine-tuning hyperparameters with the help of cross-validation and grid search.
6.  Calculating model performance using metrics such as accuracy, precision, recall, and F1-score.
7.  Generating predictions on testing data.
8.  Comparing predicted labels against the actual labels to determine model performance.

## Dataset
The Iris dataset is a well-known dataset in the field of machine learning. It contains 150 instances, each with 4 features (sepal length, sepal width, petal length, petal width) and one of three class labels (Iris-setosa, Iris-versicolor, Iris-virginica). The Iris data that been used can be download on https://tinyurl.com/4nntwy6c

## Project Steps
### 1.  Loading packages needed
```
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
```
### 2.  Create a SparkSession
```
spark = SparkSession.builder.getOrCreate()
```
### 3.  Load the iris dataset
```
iris = spark.read.csv("/user/maria_dev/irffan/Iris.csv", inferSchema=True, header=True)
iris.show()
```
