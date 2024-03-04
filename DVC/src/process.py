import numpy as np
import pandas as pd
import string
import re
import nltk
from textblob import TextBlob
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)

english_punctuations = string.punctuation
punctuations_list = english_punctuations

df = pd.read_csv('data/scrape/Tweets.csv')
# Preprocess text Tdata
def preprocess_text(text):
    text = re.sub(r"\@\w+", "", text)  # Remove usernames
    text = re.sub(r"\#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z0-9 ]+", "", text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize words
    text = [word for word in str(text).split() if word not in STOPWORDS]  # Remove stopwords
    text = ' '.join(text)  # Join tokens back into string
    text = re.sub(r'(.)1+', r'1', text) #Removed repeating chars
    text = re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', text) #Cleaning URLs
    text = text.translate(str.maketrans('', '', punctuations_list)) #Cleaning Punctuations
    text = re.sub('<[^>]*>', '', text) # Remove HTML markup
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) # Save emoticons for later appending
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
    return text

df['text'] = df['rawContent'].apply(preprocess_text)


#Apply sentiment using textblob to dataframe

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create two new columns ‘Subjectivity’ & ‘Polarity’

def getAnalysis(score):
        if score < 0:
            return "Negative"
        elif score == 0:
            return "Neutral"
        else:
            return "Positive"

#df['Target'] = list(map(lambda tweet: TextBlob(tweet), df['text']))

df['sentiment'] = df['text'].apply(getSubjectivity)
df['Polarity'] = df['text'].apply(getPolarity)
df['label'] = df['Polarity'].apply(getAnalysis)
#print (df.sample(5))
df.to_csv('data/processed/PTweets.csv')