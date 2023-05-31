# pylint: skip-file
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import gensim.downloader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')
stopwords = stopwords.words('english')
punct = list(punctuation)
stemmer = PorterStemmer()

def tokenize(text):
  text = text.replace("'", "")
  token_arr = word_tokenize(text.lower())
  stemmed = stem_tokens(token_arr)
  return remove_stop_and_punct(stemmed)

def stem_tokens(token_arr):
  return [stemmer.stem(token) for token in token_arr]

def remove_stop_and_punct(token_arr):
  return [token for token in token_arr if token not in stopwords and token not in punct]

def get_tf_idf(title_arr):
  """Returns a csr matrix with tf-idf vectorized titles"""
  # Add empty string to missing titles
  title_arr = title_arr.fillna("empty")
  clean_tokens = [tokenize(i) for i in title_arr]
  clean_text = [' '.join(sub_list) for sub_list in clean_tokens]
  vectorizer = TfidfVectorizer(stop_words='english')
  csr = vectorizer.fit_transform(clean_text)
  return csr

def get_w2v(title_arr):
  title_arr = title_arr.fillna("empty")
  clean_tokens = [tokenize(i) for i in title_arr]
  vectorizer = Word2Vec(clean_tokens)
  return vectorizer, clean_tokens

def get_pretrained_w2v(title_arr):
  title_arr = title_arr.fillna("empty")
  model = gensim.downloader.load('glove-twitter-100')
  # Clean title_tokens
  def_vect = np.zeros(model.vector_size)
  title_arr = title_arr.fillna("empty")
  clean_tokens = [tokenize(i) for i in title_arr]
  title_mat = np.zeros((len(title_arr), model.vector_size))
  for idx, title in enumerate(clean_tokens):
    # title_vec = def_vect
    for token in title:
      title_mat[idx] += model[token] if token in model else def_vect
      

  return model, title_mat

if __name__ == '__main__':
  df = pd.read_csv('../../data/steam-200k.csv')
  df = df.iloc[:, [0,1,3]]
  df.columns.values[0:3] =['user_id', 'title', 'rating']
  encoder = LabelEncoder()
  df['user_id'] = encoder.fit_transform(df['user_id'])
  df['item_id'] = encoder.fit_transform(df['title'])
  vec2 = get_w2v(df['title'])
  title_tokens = [tokenize(i) for i in df['title']]