from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


app = Flask(__name__)

Psy = pd.read_csv("Youtube01-Psy.csv")
Katy = pd.read_csv("Youtube02-KatyPerry.csv")
LMFAO = pd.read_csv("Youtube03-LMFAO.csv")
Eminem = pd.read_csv("Youtube04-Eminem.csv")
Shakira = pd.read_csv("Youtube05-Shakira.csv")

data= pd.concat([Psy,Katy,LMFAO,Eminem,Shakira])
data.drop(["COMMENT_ID", "DATE", "AUTHOR"],axis=1,inplace=True)

x=data['CONTENT'].values
y=data.CLASS
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

model=LinearSVC()

pipeline = make_pipeline(
    TfidfVectorizer(), 
    LinearSVC()        
)

# Fit the model
pipeline.fit(xtrain, ytrain)
classifier = LogisticRegression()
def isspam(input):
  predictions = pipeline.predict(pd.Series({'content':f'{input}'}))
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