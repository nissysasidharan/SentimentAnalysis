from train import X_test_tfidf,y_test,clf,vectorizer,clf
from sklearn.metrics import accuracy_score
import pickle
from joblib import load
import json
from pathlib import Path


# y_pred = clf.predict(sample_tweet)
# print("test result",y_pred


# Evaluate model performance
# Predict on testing data
y_pred = clf.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)



def main(repo_path):
    #test_csv_path = repo_path / "data/raw/test.csv"
    #test_data, labels = load_data(test_csv_path)
    #model = open("model/trained_model.joblib")
    #predictions = model.predict(test_data)
    #y_pred = clf.predict(X_test_tfidf)
    #model = load(repo_path / "model/trained_model.joblib")
    #predictions = model.predict(X_test_tfidf)
    #accuracy = accuracy_score(y_test, predictions)
    ##accuracy = accuracy_score(labels, predictions)
    #metrics = {"accuracy": accuracy}
    metrics = {"accuracy": accuracy}
    accuracy_path = repo_path / "metric/accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))
    print('Accuracy:', accuracy)
if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)

