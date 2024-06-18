# Iris : Classification Using Spark MLlib
## Introduction 
The Iris dataset is a well-known and widely used dataset in the field of machine learning and statistics. It consists of 150 samples of iris flowers, each described by four features: sepal length, sepal width, petal length, and petal width. The dataset is divided into three classes, each representing a different species of iris which are Iris setosa, Iris versicolor, and Iris virginica. This dataset is often used for testing and benchmarking classification algorithms due to its simplicity and the clear distinction between the classes.

In this assignment, leverage Apache Spark's MLlib, a powerful machine learning library, to perform classification on the Iris dataset. Spark MLlib provides scalable machine learning algorithms and tools, making it suitable for handling large datasets and complex computations. The goal is to build a classification model that accurately predicts the species of an iris flower based on its features.

![Alt text](https://github.com/irffanhaziq/DataManagementAssignment3/blob/main/Screenshot%202024-06-13%20115550.png)
## Project Summary

This project involves the classification tasks on the Iris dataset using Spark MLlib. The key steps covered in this project include:

1.  Loading packages needed
2.  Create a SparkSession
3.  Load the iris dataset
4.  Convert the Spark DataFrame to a pandas DataFrame.
5.  Convert the Species column to a numerical format.
6.  Create a random forest classifier.
7.  Define the hyperparameter grid.
8.  Perform grid search with cross-validation.
9.  Print the best hyperparameters and the corresponding accuracy
10.  Train the model with the best hyperparameters
11.  Calculate the accuracy of the model
12.  Calculate the precision of the model
13.  Calculate the recall of the model
14.  Calculate the F1 score of the model
15.  Create a DataFrame to show y_test and y_pred side by side

## Dataset
The Iris dataset is a well-known dataset in the field of machine learning. It contains 150 instances, each with 4 features (sepal length, sepal width, petal length, petal width) and one of three class labels (Iris-setosa, Iris-versicolor, Iris-virginica). The Iris data that been used can be download on https://tinyurl.com/4nntwy6c

## Project Steps
### Loading packages needed
```
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
```
### Create a SparkSession
```
spark = SparkSession.builder.getOrCreate()
```
### Load the iris dataset
```
iris = spark.read.csv("/user/maria_dev/irffan/Iris.csv", inferSchema=True, header=True)
iris.show()
```
![Alt text](https://github.com/irffanhaziq/DataManagementAssignment3/blob/main/Screenshot%202024-06-13%20115550.png)
### Convert the Spark DataFrame to a pandas DataFrame
```
iris_pdf = iris.toPandas()
```

### Convert the Species column to a numerical format
```
iris_pdf['Species'] = pd.factorize(iris_pdf['Species'])[0]

print(iris_pdf.head())
```
![Alt text](https://github.com/irffanhaziq/DataManagementAssignment3/blob/main/Screenshot%202024-06-13%20t115550.png)
```
X = iris_pdf.drop(['Id', 'Species'], axis=1)
y = iris_pdf['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Create a random forest classifier
```
clf = RandomForestClassifier()
```

### Define the hyperparameter grid
```
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
```

### Perform grid search with cross-validation
```
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

### Print the best hyperparameters and the corresponding accuracy
```
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
```
('Best Hyperparameters:', {'min_samples_split': 10, 'n_estimators': 50, 'max_dep                                     th': 15, 'min_samples_leaf': 10})

('Best Accuracy:', 0.9666666666666667)

### Train the model with the best hyperparameters
```
best_clf = RandomForestClassifier(**grid_search.best_params_)
best_clf.fit(X_train, y_train)
```

### Make predictions on the test data
```
y_pred = best_clf.predict(X_test)
```

### Calculate the accuracy of the model
```
accuracy = accuracy_score(y_test, y_pred)
print("ACCURACY OF THE MODEL:", accuracy)
```
('ACCURACY OF THE MODEL:', 0.9666666666666667)

### Calculate the precision of the model
```
precision = precision_score(y_test, y_pred, average='weighted')
print("PRECISION OF THE MODEL:", precision)
```
('PRECISION OF THE MODEL:', 0.9714285714285714)
### Calculate the recall of the model
```
recall = recall_score(y_test, y_pred, average='weighted')
print("RECALL OF THE MODEL:", recall)
```
('RECALL OF THE MODEL:', 0.9666666666666667)
### Calculate the F1 score of the model
```
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 SCORE OF THE MODEL:", f1)
```
('F1 SCORE OF THE MODEL:', 0.9672820512820512)
### Create a DataFrame to show y_test and y_pred side by side
```
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(results_df)
spark.stop()
```
![Alt text](https://github.com/irffanhaziq/DataManagementAssignment3/blob/main/Screenshot%202024-06-13%20150917.png)

