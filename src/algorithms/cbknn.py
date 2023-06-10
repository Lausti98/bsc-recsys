# pylint: skip-file
import numpy as np
import pandas as pd

from src.feature_extraction.vectorize import get_tf_idf, get_w2v

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from daisy.model.KNNCFRecommender import convert_df

class TFIDFKNN:
  def __init__(self, config):
    self.K = config['topk']
    self.maxk = config['maxk']
    self.title_col = config['title_col']
    self.user_num = config['user_num']
    self.item_num = config['item_num']

    self.model = NearestNeighbors(n_neighbors=self.K, metric='cosine')

  def fit(self, X):
    # Get user-item interaction matrix
    interactions = convert_df(self.user_num, self.item_num, X).T
    self.interactions = interactions

    # find unique items
    _items = X.drop_duplicates(subset='item')
    unique_items = pd.DataFrame({'item': range(self.item_num)})
    unique_items = unique_items.merge(_items, on='item', how='outer')
    
    self.title_mat = get_tf_idf(unique_items[self.title_col])

  
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
  
  def predict(self, user, item=None):
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
    user_embedding = self.get_user_embedding(items_rated_by_user, user)

    if item:
      return cosine_similarity(user_embedding, self.title_mat[item])
    else:
      return cosine_similarity(user_embedding, self.title_mat)

  
  def rank(self, test_df):
    t_users = test_df['user'].unique()
    ranks = []
    if len(t_users) > 1:
      for user in t_users:
        ranks.append(self.full_rank(user))
    else:
      ranks.append(self.full_rank(int(test_df['user'])))
    return np.array(ranks)


  def full_rank(self, user):
    # Find user embeddings
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
    user_embedding = self.get_user_embedding(items_rated_by_user, user)

    similarities, neighbors = self.get_neighbors(user_embedding)

    # Filter items already rated by use
    seen_items = np.in1d(neighbors, items_rated_by_user).nonzero()[0]
    similarities = np.array(np.delete(similarities, seen_items))
    neighbors = np.array(np.delete(neighbors, seen_items))

    # Find neighbouring items
    index_arr = np.argsort(similarities).flatten()[::-1]
    neighbors = neighbors[index_arr[:self.K]]

    return neighbors
  
  
  def get_user_embedding(self, items_rated_by_user, user):
    if len(items_rated_by_user) == 0:
      items_rated_by_user = np.array([0])
      user_embedding = np.zeros(self.title_mat.shape[1])
    else:
      item_vectors = self.title_mat[items_rated_by_user].A
      user_ratings = self.interactions[items_rated_by_user, user].A.squeeze()

      if user_ratings.size == 1:
        user_ratings = np.array([user_ratings])
      
      user_embedding = item_vectors * user_ratings[:, np.newaxis]
      user_embedding = np.sum(user_embedding, axis=0)
      user_embedding = user_embedding / np.sum(user_ratings)
    
    return user_embedding.reshape(1,-1)
  
  def __str__(self) -> str:
    return f'content-based TF-IDF(K={self.K})'


class Word2VecKNN:
  def __init__(self, config) -> None:
    self.K = config['topk']
    self.maxk = config['maxk']
    self.title_col = config['title_col']
    self.user_num = config['user_num']
    self.item_num = config['item_num']
  
  def fit(self, X):
    _items = X.drop_duplicates(subset='item')
    unique_items = pd.DataFrame({'item': range(self.item_num)})
    
    unique_items = unique_items.merge(_items, on='item', how='outer')
    unique_items.fillna("empty")
    unique_items.to_csv('unique_items.csv')

    # fit interaction matrix
    interactions = convert_df(self.user_num, self.item_num, X).T
    self.interactions = interactions

    
    self.w2v_model, self.title_mat = get_w2v(unique_items[self.title_col])


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
  
  def predict(self, user, item=None):
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
    user_embedding = self.get_user_embedding(items_rated_by_user, user)

    if item:
      return cosine_similarity(user_embedding, self.title_mat[item])
    else:
      return cosine_similarity(user_embedding, self.title_mat)


  def rank(self, test_df):
    t_users = test_df['user'].unique()
    ranks = []
    if len(t_users) > 1:
      for user in t_users:
        ranks.append(self.full_rank(user))
    else:
      ranks.append(self.full_rank(int(test_df['user'])))
    return np.array(ranks)


  def full_rank(self, user):
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
    user_embedding = self.get_user_embedding(items_rated_by_user, user)
    
    similarities, neighbors = self.get_neighbors(user_embedding)

    seen_items = np.in1d(neighbors, items_rated_by_user).nonzero()[0]
    similarities = np.array(np.delete(similarities, seen_items))
    neighbors = np.array(np.delete(neighbors, seen_items))
    
    index_arr = np.argsort(similarities).flatten()[::-1]
    neighbors = neighbors[index_arr[:self.K]]
    
    return neighbors

  
  def get_user_embedding(self, items_rated_by_user, user):
    if len(items_rated_by_user) == 0:
      items_rated_by_user = np.array([0])
      user_embedding = np.zeros(self.title_mat.shape[1])
    else:
      item_vectors = self.title_mat[items_rated_by_user]
      user_ratings = self.interactions[items_rated_by_user, user].A.squeeze()
      if user_ratings.size == 1:
        user_ratings = np.array([user_ratings])
      
      user_embedding = item_vectors * user_ratings[:, np.newaxis]
      user_embedding = np.sum(user_embedding, axis=0)
      user_embedding = user_embedding / np.sum(user_ratings)
    
    return user_embedding.reshape(1,-1)
  
  def _get_num_similar(self, user):
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
    user_embedding = self.get_user_embedding(items_rated_by_user, user)
    
    similarities, _ = self.get_neighbors(user_embedding)
    
    return np.count_nonzero(similarities)

  def __str__(self) -> str:
    return f'content-based Word2Vec (K={self.K})'