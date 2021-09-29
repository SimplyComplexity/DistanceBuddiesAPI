import numpy as np
import pandas as pd
from sklearn import tree

input_file = "/Users/andyyeung/Downloads/NoSpace.csv"
df = pd.read_csv(input_file, header = 0)
df.columns.str.replace(' ','')


df.head()
Answers = [(df.iloc[:0,9].values)]
Q1 = [(df.iloc[:,1].values)]
Q2 = [(df.iloc[:,2].values)]
Q3 = (df.iloc[:,3].values)
Q4 = (df.iloc[:,4].values)
Q5 = (df.iloc[:,5].values)
Q6 = (df.iloc[:,6].values)
Q7 = (df.iloc[:,7].values)
Q8 = (df.iloc[:,8].values)
Q9 = (df.iloc[:,9].values)

StudentSkillLevel = (df.iloc[:,10].values)


AverageScore = [(df.iloc[:,12].values)]




print(Answers)
print(AverageScore)

features = list(df.columns[:9])
print(features)


df.head()


X = df["AverageScore"]
y = df[features]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X,y)

clf.predict(0.7)