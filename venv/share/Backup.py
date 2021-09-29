import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ONLY KEEP FOR DEMO
import pandas as pd
from sklearn import tree
import csv as csv
input_file = "/Users/andyyeung/Downloads/AnswerData.csv"
df = pd.read_csv(input_file, header = 0)
WhichStudent = int(input("Student?"))

d = {'Y': 1, 'N': 0}
df["FACTUAL"] = df["FACTUAL"].map(d)
df["NEEDS CALC"] = df["NEEDS CALC"].map(d)
df["CORR OR NOT"] = df["CORR OR NOT"].map(d)
df.head()
features = list(df.columns[:4])
print(features)


with open('/Users/andyyeung/Downloads/Label2.csv', newline='') as csvfile:
    temp_reader = csv.reader(csvfile, delimiter=',')
    data = list(temp_reader)
row_val, col_val = WhichStudent, 4
try:

    studentSkillLevel = int((data[row_val][col_val]))
    print(studentSkillLevel)
except IndexError:
    print('No data found')


studentTrueLevel = studentSkillLevel/20


df.head()
x = df[features]
y = df["CORR OR NOT"]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

from IPython.display import Image
from sklearn.externals.six import StringIO

#easiest
print (clf.predict([[studentTrueLevel,1,1,1]]))
print (clf.predict([[studentTrueLevel,0,1,0]]))
#easy
print (clf.predict([[studentTrueLevel,1,2,5]]))
print (clf.predict([[studentTrueLevel,0,2,0]]))
#average
print (clf.predict([[studentTrueLevel,1,3,1]]))
print (clf.predict([[studentTrueLevel,1,3,0]]))
#hard
print (clf.predict([[studentTrueLevel,0,4,0]]))
print (clf.predict([[studentTrueLevel,1,4,1]]))
#hardest
print (clf.predict([[3,0,5,0]]))
print (clf.predict([[3,1,5,1]]))

