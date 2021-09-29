import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ONLY KEEP FOR DEMO
import pandas as pd
from sklearn import tree
import csv as csv

from IPython.display import Image
from sklearn.externals.six import StringIO

#dataframe filter by column value: https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values




input_file = "/Users/andyyeung/Downloads/question_bank_1.csv"
df = pd.read_csv(input_file)

questionatmp_file = "/Users/andyyeung/Downloads/question_attempt_1.csv"
questionatmp = pd.read_csv(questionatmp_file)

print(df.dtypes)
print(questionatmp.dtypes)



combined = pd.merge(questionatmp,df, how='outer', on='question_id')
print(combined.dtypes)
combined = combined.dropna()

print(combined.head())

#People to demo the algorithim
#s2018123@stu.ssc.edu.hk (top student) (show ch.1 and 2)
#s2018172@stu.ssc.edu.hk (show ch.2 and 3)
#s2018215@stu.ssc.edu.hk (sadly, he is the worst) (show ch.1)


combined_new = combined[combined['User'] == 's2018172@stu.ssc.edu.hk']
combined_new = combined_new[['User','question_id','chapter_id','difficulty', 'Correct']]

combined_new['difficulty'] = combined_new['difficulty'].astype('float32')
combined_new['Correct'] = combined_new['Correct'].astype('float32')





#combined_new.to_csv("/Users/andyyeung/Downloads/CombinedData.csv")
x = combined_new[['chapter_id','difficulty']]

print(x)

y = combined_new['Correct']



#print(y.head())
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)



print(clf.predict(1,1))

