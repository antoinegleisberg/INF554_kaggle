import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
from nltk.corpus import stopwords 

# Load the training data
train_data = pd.read_csv("train.csv")
eval_data = pd.read_csv("evaluation.csv")

# Transform our data into tfidf vectors
vectorizer = TfidfVectorizer(max_features=100, stop_words=stopwords.words('french'))
y_train = train_data['retweets_count']
X_train = vectorizer.fit_transform(train_data['text'])
# We fit our model using the training data


# A dummy regressor that always predicts the mean value of the training data
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
dummy_pred = dummy_regr.predict(X_val)
with open("mean_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "retweets_count"])
    for index, prediction in enumerate(dummy_pred):
        writer.writerow([str(eval_data['TweetID'].iloc[index]) , str(int(prediction))])

# A dummy regressor that always predicts "0" retweets
dummy_regr = DummyRegressor(strategy="constant", constant=0)
dummy_regr.fit(X_train, y_train)
dummy_pred = dummy_regr.predict(X_val)
with open("zero_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "retweets_count"])
    for index, prediction in enumerate(dummy_pred):
        writer.writerow([str(eval_data['TweetID'].iloc[index]) , str(int(prediction))])


