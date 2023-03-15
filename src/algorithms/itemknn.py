# pylint: skip-file
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix

class ItemKNN:
  def __init__(self, K=10):
    self.K = K
    self.model = NearestNeighbors(n_neighbors=K, metric='cosine')

  def fit(self, X):
    #TODO: Do checks on the input DF/Interactionmatrix

    self.data = X
    #self.model.fit(X)
  
  def get_neighbors(self, X, verbose=False):
    """Get k closest neighbors by their cosine similarity"""
    # Find similarities
    similarities = cosine_similarity(X, self.data)
    index_arr = np.argsort(similarities, axis=1).flatten()[::-1] #reverse sorted array for descending order

    neighbors = index_arr[:self.K]
    distances = 1 - np.take(similarities, neighbors) # subtract 1 to make distance 0 optimal
    
    if verbose:
      for i in range(len(distances)):
        print(f'Item {i+1}:  Distance - {distances[i]}, item ID - {neighbors[i]}')
    return distances, neighbors
  
  def predict(self, X):
    #TODO: Implement actual predictions 
    _, neighbors = self.get_neighbors(X)
    return neighbors