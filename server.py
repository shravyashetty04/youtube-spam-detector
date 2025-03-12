from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


app = Flask(__name__)

Psy = pd.read_csv("Youtube01-Psy.csv")
Katy = pd.read_csv("Youtube02-KatyPerry.csv")
LMFAO = pd.read_csv("Youtube03-LMFAO.csv")
Eminem = pd.read_csv("Youtube04-Eminem.csv")
Shakira = pd.read_csv("Youtube05-Shakira.csv")
Spam = pd.read_csv("Youtube-Spam-Dataset.csv")

data = pd.concat([Psy, Katy, LMFAO, Eminem, Shakira, Spam], ignore_index=True)
print("Missing values before cleaning:", data.isna().sum())
data.dropna(inplace=True)
data.drop(["COMMENT_ID", "DATE", "AUTHOR", "VIDEO_NAME"], axis=1, inplace=True)

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

X = data['CONTENT']
y = data['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

estimators = [
    ('nb', MultinomialNB()),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=SVC(kernel='rbf', probability=True, C=1, gamma=0.1),
    cv=5
)

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', stacking_model)
])
pipeline.fit(X_train_tfidf, y_train_encoded)

def isspam(input):
    predictions = pipeline.predict(vectorizer.transform(pd.Series({'content':f'{input}'})))
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