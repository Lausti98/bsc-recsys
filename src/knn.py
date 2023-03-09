# pylint: skip-file
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


def ItemKNN(X_train, y_train, K=10):
  X_train = X_train[~X_train[['product_id', 'user_id']].duplicated()]
  # TODO: perform this operation otherhow, since pivot is very large dense matrix
  user_c = CategoricalDtype(sorted(X_train['user_id'].unique()), ordered=True)
  item_c = CategoricalDtype(sorted(X_train['product_id'].unique()), ordered=True)

  col = X_train['user_id'].astype(user_c).cat.codes
  row = X_train['product_id'].astype(item_c).cat.codes
  # print(row)
  csr = csr_matrix((X_train['overall'], (row, col)), \
                           shape=(item_c.categories.size, user_c.categories.size))
  # pivot = X_train.pivot(
  #   index='product_id',
  #   columns='user_id',
  #   values='overall'
  # ).fillna(0) # missing rating is filled as 0
  # print(pivot)
  # csr = csr_matrix(pivot.values)
  #print(csr)
  # print(csr.get_shape())
  # print(csr.getrow(0).sum())
  # print(csr.getrow(981).sum())
  #sim_mat = cosine_similarity(X_train, X_train)
  #print(sim_mat)
  model = NearestNeighbors(n_neighbors=100, metric='cosine', algorithm='brute')
  model.fit(csr)
  dist, idx = model.kneighbors(csr.getrow(140))
  for i, j in zip(dist, idx):
    print(i,j)
  

df = pd.read_csv('../data/software_rating_only.csv')
print(df['product_id'].nunique(), len(df))
# print(df.head(101))
print(df[df['product_id'] == 5])
ItemKNN(df, y_train=df['overall'].head())