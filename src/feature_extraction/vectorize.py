# pylint: skip-file
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')
stopwords = stopwords.words('english')
punct = list(punctuation)

def tokenize(text):
  text = text.replace("'", "")
  token_arr = word_tokenize(text.lower())
  return remove_stop_and_punct(token_arr)

def remove_stop_and_punct(token_arr):
  return [token for token in token_arr if token not in stopwords and token not in punct]

def get_tf_idf(title_arr):
  """Returns a csr matrix with tf-idf vectorized titles"""
  vectorizer = TfidfVectorizer(stop_words='english')
  csr = vectorizer.fit_transform(title_arr)
  return csr

def get_w2v(title_arr):
  clean_tokens = [tokenize(i) for i in title_arr]
  vectorizer = Word2Vec(clean_tokens)
  return vectorizer

if __name__ == '__main__':
  df = pd.read_csv('../../data/steam-200k.csv')
  df = df.iloc[:, [0,1,3]]
  df.columns.values[0:3] =['user_id', 'title', 'rating']
  encoder = LabelEncoder()
  df['user_id'] = encoder.fit_transform(df['user_id'])
  df['item_id'] = encoder.fit_transform(df['title'])
  #vec1 = get_tf_idf(df['title'])
  vec2 = get_w2v(df['title'])
  # print(df.dtypes)
  # print(df.shape)
  # #print(vec1)
  # print(vec2.wv.key_to_index)
  # print(vec2.wv.most_similar(['fallout']))
  # print(vec2.wv.most_similar(['team', 'fortress', '2']))
  # print(vec2.wv.most_similar(['fallout', 'new', 'vegas', 'couriers', 'stash']))
  print(vec2.wv.n_similarity(['fallout', 'new', 'vegas', 'couriers', 'stash'], ['fallout', '3']))
  print(vec2.wv.similarity('counter-strike', 'fallout'))