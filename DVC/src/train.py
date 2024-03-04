import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from joblib import dump
from pathlib import Path
import yaml
import pickle
import json
import re
from sklearn.metrics import accuracy_score
import csv

data = pd.read_csv('data/processed/PTweets.csv')
#data_path = repo_path / "data"
#train_path = data_path / "raw/train"
#test_path = data_path / "raw/val"
#train_files, train_labels = get_files_and_labels(train_path)
#test_files, test_labels = get_files_and_labels(test_path)
#prepared = data_path / "prepared"

params = yaml.safe_load(open("params.yaml"))["train"]
split = params["split"]
random = params["random"]
# Split Tdata into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['rawContent'], data['label'], test_size=split, random_state=random)
#save_as_csv(X_train, y_train, raw/train/ "train.csv")
#save_as_csv(X_test, y_test, raw/val/ "test.csv")

# Vectorize text Tdata using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a linear support vector machine (SVM) classifier
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

# Predict on testing data
#y_pred = clf.predict(X_test_tfidf)

# Evaluate model performance
#accuracy = accuracy_score(y_test, y_pred)
#print('Accuracy:', accuracy)



    #test_csv_path = repo_path / "data/raw/test.csv"
    #test_data, labels = load_data(test_csv_path)
    #model = open("model/trained_model.joblib")
    #predictions = model.predict(test_data)
    #y_pred = clf.predict(X_test_tfidf)

    #print('Accuracy:', accuracy)

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(clf, open('classifier.pkl', 'wb'))


    #predictions = model.predict(X_test_tfidf)
    #accuracy = accuracy_score(y_test, predictions)
    #accuracy = accuracy_score(labels, predictions)
    #metrics = {"accuracy": accuracy}
    #metrics = {"accuracy": accuracy}
    #accuracy_path = repo_path / "metric/accuracy.json"
    #accuracy_path.write_text(json.dumps(metrics))

