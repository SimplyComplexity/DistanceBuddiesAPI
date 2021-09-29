from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask import request
import pandas as pd
import ast
import numpy as np
import warnings
import random
from sklearn import tree
import csv as csv
from IPython.display import Image
from sklearn.externals.six import StringIO
from pandas import DataFrame


#People to demo the algorithim
#s2018123@stu.ssc.edu.hk (top student) (show ch.1 and 2)
#s2018172@stu.ssc.edu.hk (show ch.2 and 3)
#s2018215@stu.ssc.edu.hk (sadly, he is the worst) (show ch.1)

print("fuck you")


app = Flask(__name__)
api = Api(app)

#api
class question_predict(Resource):
    # methods go here

    def get(self):

        #params that the user sends
        user = request.args.get('user')
        chapter = request.args.get('chapter')



        #change these to the server files soon
        questionbank_file = "/Users/andyyeung/Downloads/question_bank_4.csv"
        questionbank = pd.read_csv(questionbank_file)




        questionatmp_file = "/Users/andyyeung/Downloads/question_attempt_1.csv"
        questionatmp = pd.read_csv(questionatmp_file)





        combined = pd.merge(questionatmp,questionbank, how='outer', on='question_id')


        combined_new = combined[combined['User'] == user]
        combined_new = combined_new[['User', 'question_id', 'chapter_id', 'difficulty', 'Correct','param_for_ai_diagram','param_for_ai_calc']]





        combined_new['chapter_id'] = combined_new['chapter_id'].astype('float32')
        combined_new['difficulty'] = combined_new['difficulty'].astype('float32')
        combined_new['Correct'] = combined_new['Correct'].astype('float32')
        combined_new['param_for_ai_diagram'] = combined_new['Correct'].astype('float32')
        combined_new['param_for_ai_calc'] = combined_new['Correct'].astype('float32')


        #checks for the maximum chapter, if you enter a value above the chapters you can choose it will just give you questions from every chapter
        column = combined_new['chapter_id']
        max_chapter = column.max()
        print(max_chapter)

        if(float(chapter)) < max_chapter:
            combined_new = combined_new[combined_new['chapter_id'] == float(chapter)]

        x = combined_new[['chapter_id', 'difficulty','param_for_ai_diagram','param_for_ai_calc']]
        y = combined_new['Correct']


        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x, y)

        index = questionbank.index
        number_of_rows = len(index)


        currentValue = 1

        post_predictionlist = []

        for question in range (1,number_of_rows):


            #convert into correct prediction format


            prediction_df_old = (questionbank.iloc[currentValue])
            prediction_df_new = prediction_df_old[['chapter_id', 'difficulty','param_for_ai_diagram','param_for_ai_calc']]
            predictionlist = prediction_df_new.values.tolist()


            #prediction
            predictionResult = (clf.predict([predictionlist]))
            currentValue += 1

            if predictionResult == 0:

                questionValue = currentValue
                #add the result to the list
                post_predictionlist.append(questionValue)

            if currentValue == number_of_rows:
                break


                #generating questions



        if len(post_predictionlist) >= 10:
            finalQuestion = random.sample(post_predictionlist, 10)
        else:
            finalQuestion = random.sample(post_predictionlist, len(post_predictionlist))
            questionsLacking = 10 - post_predictionlist
            finalQuestion.append(random.sample(range(1, number_of_rows), questionsLacking))


        predictionResultsDataframe = DataFrame(finalQuestion, columns=['Prediction'])

        data = predictionResultsDataframe
        data = data.to_dict()  # convert dataframe to dictionary
        return {'data': data}, 200  # return data and 200 OK code


    pass

api.add_resource(question_predict, '/question_predict')  # '/question_attempt' is our entry point for Question Attempt

if __name__ == '__main__':
    app.run()  # run our Flask app