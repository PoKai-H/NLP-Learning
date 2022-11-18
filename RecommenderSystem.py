import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



class Recommender:
      
    def read_csv(self):
        df = pd.read_csv(r'C:\python\NLP_machine-learning\tmdb_5000_movies.csv')
        return df
    def genres_and_keywords_to_string(self,row):
        genres = json.loads(row['genres'])
        genres = ' '.join(''.join(j['name'].split()) for j in genres)

        keywords = json.loads(row['keywords'])
        keywords = ' '.join(''.join(i['name'].split())for i in keywords)
        return f'{genres} {keywords}'
    def apply_to_string_func(self):
        df = self.read_csv()
        df['string'] = df.apply(self.genres_and_keywords_to_string, axis=1)
        labels = df['string']
        return labels
    def tf_idf(self):
        labels = self.apply_to_string_func()
        tfidf = TfidfVectorizer(max_features=2000)
        X = tfidf.fit_transform(labels)
        return X
    def movie2idx(self):
        df = self.read_csv()
        movie_to_idx = pd.Series(df.index, index=df['title'])
        return movie_to_idx
    def recommand(self,title):
        # get the row in the dataframe for this movie
        df = self.read_csv()
        movie2idx = self.movie2idx()
        idx = movie2idx[title]
        X = self.tf_idf()
        if type(idx) == pd.Series: # if the movie title is the same for mutiple rows, it will return a pandas.Series rather than index
            idx = idx.iloc[0] # we grep the first item instead
        query = X[idx]
        scores = cosine_similarity(query, X)

        scores = scores.flatten()
        recommended_idx = (-scores).argsort()[1:6]

        return df['title'].iloc[recommended_idx]

recommand = Recommender()
movie = input('Please input a Movie name :')
print(f'Recommendtions for {movie} :')
print(recommand.recommand(movie))