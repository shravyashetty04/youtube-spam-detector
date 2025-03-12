from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

Psy = pd.read_csv("Youtube01-Psy.csv")
Katy = pd.read_csv("Youtube02-KatyPerry.csv")
LMFAO = pd.read_csv("Youtube03-LMFAO.csv")
Eminem = pd.read_csv("Youtube04-Eminem.csv")
Shakira = pd.read_csv("Youtube05-Shakira.csv")

data= pd.concat([Psy,Katy,LMFAO,Eminem,Shakira])
data.drop(["COMMENT_ID", "DATE", "AUTHOR"],axis=1,inplace=True)
xtrain,xtest,ytrain,ytest=train_test_split(data["CONTENT"],data["CLASS"])
tfidf_vect = TfidfVectorizer(use_idf=True, lowercase=True)
xtrain_tfidf = tfidf_vect.fit_transform(xtrain)
model = MultinomialNB()
model.fit(xtrain_tfidf, ytrain)

def isspam(input):
    xtest_tfidf = tfidf_vect.transform(pd.Series({'CONTENT':input}))
    predictions = model.predict(xtest_tfidf)
    if predictions[0]==1:
        return True
    else:
        return False

@app.route('/api/<input>')
def hello_world(input):
    result=isspam(input)
    return jsonify(result=result)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()