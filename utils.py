import pandas as pd
from datetime import datetime
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords

from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer

nltk.download("stopwords")


def feature_engineering(input_df: pd.DataFrame) -> pd.DataFrame:
    res_df = input_df

    # add a column to data which counts url
    res_df["url_count"] = res_df["urls"].str.count("http")
    # add columns to data which makes ratios
    res_df["followers_friends"] = res_df["followers_count"] / (res_df["friends_count"].apply(lambda x: x + 1))

    return res_df


def time_engineering(input_df: pd.DataFrame) -> pd.DataFrame:
    res_df = input_df
    res_df["hour"] = res_df["timestamp"].apply(lambda t: (datetime.fromtimestamp(t // 1000).hour))
    res_df["day"] = res_df["timestamp"].apply(lambda t: (datetime.fromtimestamp(t // 1000)).weekday())
    res_df["week_in_month"] = res_df["timestamp"].apply(lambda t: (datetime.fromtimestamp(t // 1000).day) // 7)

    return res_df


def extract_topic(input_df: pd.DataFrame) -> pd.DataFrame:
    res_df = input_df
    res_df["hashtags"] = res_df["hashtags"].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", ""))
    # join text and hashtags
    res_df["total_text"] = res_df["text"] + " " + res_df["hashtags"]
    vectorizer = TfidfVectorizer(min_df=1, max_features=None, stop_words=stopwords.words("french"))
    vector = vectorizer.fit_transform(res_df["total_text"])
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    svd.fit(vector)
    topic = svd.transform(vector)
    res_df["topic_1"] = topic[:, 0]
    res_df["topic_2"] = topic[:, 1]
    res_df["topic_3"] = topic[:, 2]
    res_df["topic_4"] = topic[:, 3]
    res_df["topic_5"] = topic[:, 4]
    return res_df


def text_engineering(input_df: pd.DataFrame) -> pd.DataFrame:
    res_df = input_df
    # add columns related to sentiment analysis
    res_df["polarity"] = res_df["total_text"].apply(
        lambda x: TextBlob(x, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0]
    )
    res_df["subjectivity"] = res_df["total_text"].apply(
        lambda x: TextBlob(x, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[1]
    )

    return res_df


def hashtags_engineering(input_df):
    res_df = input_df

    # add a column to data which gives number of hashtags
    res_df["hashtags_count"] = res_df["hashtags"].apply(
        lambda hashtags: len(hashtags.split(",")) if hashtags != "" else 0
    )

    return res_df


def extract_cluster(input_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    res_df = input_df
    res_df["cluster"] = KMeans(n_clusters=100, random_state=0).fit_predict(res_df[columns].values)
    return res_df


def clean_data(input_df: pd.DataFrame, reindex_cols: List[str], remove_cols: List[str]) -> pd.DataFrame:
    res_df = input_df
    res_df = feature_engineering(res_df)
    res_df = time_engineering(res_df)
    res_df = extract_topic(res_df)
    res_df = text_engineering(res_df)
    res_df = hashtags_engineering(res_df)
    res_df = extract_cluster(
        res_df,
        [
            "followers_count",
            "friends_count",
            "favorites_count",
            "statuses_count",
            "followers_friends",
            "polarity",
            "subjectivity",
            "hashtags_count",
        ],
    )
    res_df = res_df.reindex(columns=reindex_cols)
    res_df = res_df.drop(remove_cols, axis=1)
    return res_df
