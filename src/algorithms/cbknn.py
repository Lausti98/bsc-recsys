# pylint: skip-file
import numpy as np
import pandas as pd

from src.feature_extraction.vectorize import get_tf_idf, get_w2v, get_pretrained_w2v

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from daisy.model.KNNCFRecommender import convert_df
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix

class TFIDFKNN:
  def __init__(self, config):
    self.K = config['topk']
    self.maxk = config['maxk']
    self.model = NearestNeighbors(n_neighbors=self.K, metric='cosine')
    self.title_col = config['title_col']
    self.user_num = config['user_num']
    self.item_num = config['item_num']
    # self.vectorizer = config['vectorizer']

  def fit(self, X):
    # Use the title column and vectorize
    ### 
    # 1. Get items the user liked / disliked 
    # 2. Get similarities of items the user liked in the past
    interactions = convert_df(self.user_num, self.item_num, X).T
    _items = X.drop_duplicates(subset='item')
    unique_items = pd.DataFrame({'item': range(self.item_num)})
    
    unique_items = unique_items.merge(_items, on='item', how='outer')
    
    # unique_items = X.drop_duplicates(subset='item')
    self.title_mat = get_tf_idf(unique_items[self.title_col])


    # prediction matrix
    # self.pred_mat = cosine_similarity(X, X, dense_output=False).dot(X)
    # print(self.pred_mat)s
    # print(self.pred_mat.shape)
    self.interactions = interactions

    # X = convert_df(self.user_num, self.item_num, X).T

    # # Find mean user rating 
    # sums = X.sum(axis=0)
    # non_zeros = np.count_nonzero(X.toarray(), axis=0)
    # means = sums.flatten() / (non_zeros+(non_zeros==0)) # Account for possible divide by 0

    # # Adjust ratings by means
    # X_transpose = X.transpose() 
    # nz = X_transpose.nonzero()
    # X_transpose = X_transpose.astype('float64')
    # X_transpose[nz] -= means[0, nz[0]]

    # # Assign mean adjusted data to self
    # self.data = X_transpose.transpose()
    # self.user_offset = means[0,:]

    # # prediction matrix
    # self.pred_mat = cosine_similarity(self.data, self.data, dense_output=False).dot(X)

  
  def get_neighbors(self, X, verbose=False):
    """Get k closest neighbors by their cosine similarity"""

    # Find similarities
    similarities = cosine_similarity(X, self.title_mat)
    index_arr = np.argsort(similarities, axis=1).flatten()[::-1] #reverse sorted array for descending order

    # Get K neighbors from similarities
    neighbors = index_arr[:self.maxk]
    distances = np.take(similarities, neighbors)
    
    if verbose:
      for i in range(len(distances)):
        print(f'Item {i+1}:  Distance - {distances[i]}, item ID - {neighbors[i]}')
    

    return distances, neighbors
  
  def predict(self, X, user):

    # Get the nearest neighbors
    dist, neighbors = self.get_neighbors(X)

    # Calculate prediction based on neighbors
    rating = 0
    for d, n in zip(dist, neighbors):
      rating += self.data[n,user]*d
    prediction = (rating/(dist.sum()+(dist.sum()==0.0))) + self.user_offset[0, user]

    return prediction if prediction != None else 0.0

  def pred_from_mat(self, user, item):
    return self.pred_mat[item, user]
  
  def rank(self, test_df):
    ranks = []
    if len(test_df) > 1:
      for _, row in test_df.iterrows():
        ranks.append(self.full_rank(int(row['user'])))
    else:
      ranks.append(self.full_rank(int(test_df['user'])))
    return np.array(ranks)


  def full_rank(self, user):
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
    # print(items_rated_by_user)
    if len(items_rated_by_user) == 0:
      items_rated_by_user = np.array([0])
    similarities = []
    neighbors = []
    # print(self.get_neighbors(self.title_mat[items_rated_by_user, :]))
    for i in items_rated_by_user:
      sim, neigh = self.get_neighbors(self.title_mat[i])
      similarities.append(sim)
      neighbors.append(neigh)

    similarities = [item for sublist in similarities for item in sublist]
    neighbors = [item for sublist in neighbors for item in sublist]
    seen_items = np.in1d(neighbors, items_rated_by_user).nonzero()[0]
    similarities = np.array(np.delete(similarities, seen_items))
    neighbors = np.array(np.delete(neighbors, seen_items))

    index_arr = np.argsort(similarities).flatten()[::-1]
    neighbors = neighbors[index_arr[:self.K]]
    return neighbors
  
  def __str__(self) -> str:
    return f'content based TF-IDF KNN(K={self.K})'


class Word2VecKNN:
  def __init__(self, config, pretrained=False) -> None:
    self.K = config['topk']
    self.maxk = config['maxk']
    self.title_col = config['title_col']
    self.user_num = config['user_num']
    self.item_num = config['item_num']
    self.pretrained = pretrained
  
  def fit(self, X):
    _items = X.drop_duplicates(subset='item')
    unique_items = pd.DataFrame({'item': range(self.item_num)})
    
    unique_items = unique_items.merge(_items, on='item', how='outer')
    print(unique_items.info())
    unique_items.fillna("empty")
    unique_items.to_csv('unique_items.csv')
    # print(unique_items[self.title_col])
    # print(X.dtypes)

    # unique_items = X.drop_duplicates(subset='item')
    if self.pretrained:
      self.w2v_model, self.title_tokens = get_pretrained_w2v(unique_items[self.title_col])
    else:
      self.w2v_model, self.title_tokens = get_w2v(unique_items[self.title_col])
    # print(self.title_tokens)

    self.sim_mat = np.zeros((self.item_num, self.item_num))
    for idx, title in enumerate(self.title_tokens):
      if len(title) == 0:
          # print(title)
        title = ['empty']
      self.sim_mat[:,idx] = self._get_similarities(title)
    # print(self.sim_mat)
    # fit interaction matrix
    interactions = convert_df(self.user_num, self.item_num, X).T
    self.interactions = interactions
    

  def _get_similarities(self, title_tokens):
    """Use Word2Vec model to get similarities of all items"""
    sims = []
    for title in self.title_tokens:
      if title != title_tokens:
        if len(title) == 0:
          # print(title)
          title = ['empty']
        # print(title)
        if self.pretrained:
          sim = self.w2v_model.n_similarity(title_tokens, title)
        else:
          # print(title)
          sim = self.w2v_model.wv.n_similarity(title_tokens, title)
        sims.append(sim)
      else:
        sims.append(0.0)
        # print(sim)
        #print(f'sim {title_tokens}, {title}: {sim}')
    
    return np.array(sims).reshape((1,-1))


  def get_neighbors(self, item, verbose=False):
    """Get k closest neighbors by their cosine similarity"""

    # Find similarities
    similarities = self.sim_mat[item].reshape((1,-1))
    #print(similarities)
    # similarities = self.w2v_model.wv.n_similarity(X, self.title_tokens)#, topn=self.maxk)
    # similarities1 = self._get_similarities(X)
    # similarities2  = cosine_similarity(X, self.title_tokens)
    # print(similarities)
    # print(similarities.shape)
    # similarities = cosine_similarity(X, self.title_mat)
    index_arr = np.argsort(similarities, axis=1).flatten()[::-1] #reverse sorted array for descending order
    # print(index_arr)
    # Get K neighbors from similarities
    neighbors = index_arr[:self.maxk]
    distances = np.take(similarities, neighbors)
    
    if verbose:
      for i in range(len(distances)):
        print(f'Item {i+1}:  Distance - {distances[i]}, item ID - {neighbors[i]}')
  

    return distances, neighbors

  def rank(self, test_df):
    ranks = []
    if len(test_df) > 1:
      for _, row in test_df.iterrows():
        ranks.append(self.full_rank(int(row['user'])))
    else:
      ranks.append(self.full_rank(int(test_df['user'])))
    return np.array(ranks)
  
  def full_rank(self, user):
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
    # print(items_rated_by_user)
    if len(items_rated_by_user) == 0:
      items_rated_by_user = np.array([0])
    similarities = []
    neighbors = []
    # print(self.get_neighbors(self.title_mat[items_rated_by_user, :]))
    for i in items_rated_by_user:
      sim, neigh = self.get_neighbors(i)
      similarities.append(sim)
      neighbors.append(neigh)

    similarities = [item for sublist in similarities for item in sublist]
    neighbors = [item for sublist in neighbors for item in sublist]
    seen_items = np.in1d(neighbors, items_rated_by_user).nonzero()[0]
    similarities = np.array(np.delete(similarities, seen_items))
    neighbors = np.array(np.delete(neighbors, seen_items))
    
    index_arr = np.argsort(similarities).flatten()[::-1]
    neighbors = neighbors[index_arr[:self.K]]

    
    return neighbors
  
  def __str__(self) -> str:
    return f'Word2Vec KNN(K={self.K}, pretrained={self.pretrained})'