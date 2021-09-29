import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

input_file = "/Users/andyyeung/Downloads/Label2.csv"
df = pd.read_csv(input_file, header = 0)
df.columns.str.replace(' ','')

# Store the training parameters as X
# X = df[['Previous year srore', 'AverageScore']].values
X = df[['1','2','3','4','5','6','7','8','9','10']].values
# Store the labels as y
y = df[['Label']].values.ravel()

# Transform label to integers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Print X, y values
# print(X, y)

# Seperate data to training and testing set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Train the model using decision tree
clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)

# Predict the result on the testing set with the model
# y_pred = clf.predict(X_test)

# Get how accurate is the model
# print(accuracy_score(y_test, y_pred))

# 10-fold cross validation
kfold = KFold(10, True, 1)
scores = cross_val_score(clf, X, y, cv=kfold)
avg_score = np.mean(scores)
print(avg_score)