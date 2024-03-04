from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)
# model = pickle.load(open("model.pkl","rb"))
@app.route("/")
def index():
	# data = request.get_json(force=True)
	# sent_result = model.sentiment_result([[np.array(data["exp"])]])
	# output = sent_result[0]
	output = "string"
	return jsonify(output)
#****
@app.route("/sentiment_result",methods= ["POST","GET"])

def sentiment_result():
	def preprocess_text(text):
	    # text = re.sub(r"http\S+", "", text)  # Remove URLs
	    text = re.sub(r"\@\w+", "", text)  # Remove usernames
	    text = re.sub(r"\#\w+", "", text)  # Remove hashtags
	    text = re.sub(r"[^a-zA-Z0-9 ]+", "", text)  # Remove non-alphanumeric characters
	    text = text.lower()  # Convert to lowercase

	    text = nltk.word_tokenize(text)  # Tokenize words
	    #text = [word for word in text if word not in nltk.corpus.stopwords.words('english')]  # Remove stopwords
	    text = ' '.join(text)  # Join tokens back into string
	    return text
	# if request post ... if request get...
	if request.method == 'POST':
		# Vectorize text data using TF-IDF
		vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

		# Train a linear support vector machine (SVM) classifier
		clf = pickle.load(open('classifier.pkl','rb'))
		# ****
		sample_tweet = request.form['sentiment']
		print(sample_tweet)
		# sample_tweet= request.get_json(force=True)['sentiment']
		sample_tweet = preprocess_text(sample_tweet)
		sample_tweet = vectorizer.transform([sample_tweet])

		y_pred = clf.predict(sample_tweet)
		return jsonify(y_pred[0])
	else:
		return render_template('index.html')


if __name__ == "__main__":
	try:
		app.run(host="0.0.0.0", port = 4999, debug = True)
	except:
		print('could not creating hosting server')

