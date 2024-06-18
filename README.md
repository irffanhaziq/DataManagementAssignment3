# Iris : Classification Using Spark MLlib

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

### Calculate the precision of the model
```
precision = precision_score(y_test, y_pred, average='weighted')
print("PRECISION OF THE MODEL:", precision)
```

### Calculate the recall of the model
```
recall = recall_score(y_test, y_pred, average='weighted')
print("RECALL OF THE MODEL:", recall)
```

### Calculate the F1 score of the model
```
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 SCORE OF THE MODEL:", f1)

print(y_test)
print(y_pred)
```

### Create a DataFrame to show y_test and y_pred side by side
```
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(results_df)
spark.stop()
```

