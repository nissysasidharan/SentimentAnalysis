import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
# Load data
# data = pd.read_csv('50000-Twitter_Data.csv')
data = pd.read_csv('data/raw/train/training.csv')

# Preprocess text data
def preprocess_text(text):
    # text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\@\w+", "", text)  # Remove usernames
    text = re.sub(r"\#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z0-9 ]+", "", text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    #text = nltk.word_tokenize(text)  # Tokenize words
    #text = [word for word in text if word not in nltk.corpus.stopwords.words('english')]  # Remove stopwords
    text = ' '.join(text)  # Join tokens back into string
    return text

data['text'] = data['rawContent'].apply(preprocess_text)
# Split data into training and testing sets
# data.mask(data.eq('None')).dropna(inplace=True,axis = 0)
X_train, X_test, y_train, y_test = train_test_split(data['rawContent'], data['label'], test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a linear support vector machine (SVM) classifier
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

# Predict on testing data
y_pred = clf.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# sample_tweet = 'love your beauty'
# sample_tweet = preprocess_text(sample_tweet)
# sample_tweet = vectorizer.transform([sample_tweet])
#
# y_pred = clf.predict(sample_tweet)
# print("test result",y_pred)


pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(clf,open('classifier.pkl','wb'))