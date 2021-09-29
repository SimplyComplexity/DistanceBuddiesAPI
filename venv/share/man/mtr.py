import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ONLY KEEP FOR DEMO
import pandas as pd
from sklearn import tree
import csv as csv
input_file = "/Users/andyyeung/Downloads/data.csv"
#put your own directory here, this is currently my directory
df = pd.read_csv(input_file, header = 0)




features = list(df.columns[:6])




df.head()
x = df[features]
y = df["Prize123"]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

from IPython.display import Image
from sklearn.externals.six import StringIO


#there are many people far away from central, no one is in central, you are far away from central
print (clf.predict([[2000,3000,4000,1000,5000,3]]))
