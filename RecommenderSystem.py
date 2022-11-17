import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

class Recommender:
    def __init__(self):
        self.tfidf = TfidfTransformer()
    def read_csv():
        pd.read_csv('tmdb_5000_movies.csv')
