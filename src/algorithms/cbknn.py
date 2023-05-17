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

  def fit(self, X):
    interactions = convert_df(self.user_num, self.item_num, X).T
    _items = X.drop_duplicates(subset='item')
    unique_items = pd.DataFrame({'item': range(self.item_num)})
    
    unique_items = unique_items.merge(_items, on='item', how='outer')
    
    self.title_mat = get_tf_idf(unique_items[self.title_col])

    self.interactions = interactions
  
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
    if len(items_rated_by_user) == 0:
      items_rated_by_user = np.array([0])
      user_embedding = np.zeros(self.title_mat.shape[1])
    else:
      item_vectors = self.title_mat[items_rated_by_user].A
      user_ratings = self.interactions[items_rated_by_user, user].A.squeeze()
      if user_ratings.size == 1:
        user_ratings = np.array([user_ratings])
      
      user_embedding = item_vectors * user_ratings[:, np.newaxis]
      user_embedding = np.mean(user_embedding, axis=0)

    similarities = []
    neighbors = []
    sim, neigh = self.get_neighbors(user_embedding.reshape(1,-1))
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
  
  
  # def get_user_embeddings(self, users):
  #   for user in users:
  #     items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
  #   if len(items_rated_by_user) == 0:
  #     items_rated_by_user = np.array([0])
  #   item_vectors = self.title_mat[items_rated_by_user]
  #   print(item_vectors)
  #   user_ratings = self.interactions[items_rated_by_user, user].A.squeeze()
  
  def __str__(self) -> str:
    return f'content based TF-IDF(K={self.K})'


class Word2VecKNN:
  def __init__(self, config, pretrained=False, similarity_method='cosine') -> None:
    self.K = config['topk']
    self.maxk = config['maxk']
    self.title_col = config['title_col']
    self.user_num = config['user_num']
    self.item_num = config['item_num']
    self.pretrained = pretrained
    self.similarity_method = similarity_method
  
  def fit(self, X):
    _items = X.drop_duplicates(subset='item')
    unique_items = pd.DataFrame({'item': range(self.item_num)})
    
    unique_items = unique_items.merge(_items, on='item', how='outer')
    unique_items.fillna("empty")
    unique_items.to_csv('unique_items.csv')

    # fit interaction matrix
    interactions = convert_df(self.user_num, self.item_num, X).T
    self.interactions = interactions

    # unique_items = X.drop_duplicates(subset='item')
    if self.pretrained or self.similarity_method == 'wordmover':
      self.w2v_model, self.title_tokens = get_pretrained_w2v(unique_items[self.title_col])
    else:
      self.w2v_model, self.title_tokens = get_w2v(unique_items[self.title_col])
    

  def _get_similarities(self, title_tokens):
    """Use Word2Vec model to get similarities of all items"""
    sims = []
    for title in self.title_tokens:
      if title != title_tokens:
        if len(title) == 0:
          title = ['empty']
        
        sim = self._switch_similarity_method(title_tokens, title)
        sims.append(sim)
      else:
        sims.append(0.0)
    
    return np.array(sims).reshape((1,-1))


  def get_neighbors(self, user, verbose=False):
    """Get k closest neighbors by their cosine similarity"""
    items_rated_by_user = self.interactions[:, user].A.squeeze().nonzero()[0]
  
    # Make vector from all titles rated by user
    item_vectors = [self.title_tokens[i] for i in items_rated_by_user]
    user_embedding = [token for sublist in item_vectors for token in sublist] 
  
    if len(user_embedding) == 0:
      user_embedding = ['empty']
    similarities = self._get_similarities(user_embedding)

    # Find similarities
    index_arr = np.argsort(similarities, axis=1).flatten()[::-1] #reverse sorted array for descending order

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
    
    # if len(items_rated_by_user) == 0:
    #   items_rated_by_user = np.array([0])
    #   user_embedding = np.array(['empty'])
    # else:
    #   item_vectors = self.title_tokens[items_rated_by_user].A
    #   # user_ratings = self.interactions[items_rated_by_user, user].A.squeeze()
    #   # if user_ratings.size == 1:
    #   #   user_ratings = np.array([user_ratings])
      
    #   # user_embedding = item_vectors * user_ratings[:, np.newaxis]
    #   # user_embedding = np.mean(user_embedding, axis=0)
    #   user_embedding = item_vectors.ravel()
    
    similarities = []
    neighbors = []
    # for i in items_rated_by_user:
    sim, neigh = self.get_neighbors(user)
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
  
  def _switch_similarity_method(self, title_tokens, title):
    if self.similarity_method == 'wordmover':
      sim = self.w2v_model.wmdistance(title_tokens, title)
    elif self.pretrained:
      sim = self.w2v_model.n_similarity(title_tokens, title)
    else:
      sim = self.w2v_model.wv.n_similarity(title_tokens, title)
    
    return sim
  
  def __str__(self) -> str:
    return f'Word2Vec (K={self.K}, pretrained={self.pretrained})'