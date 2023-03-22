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
    # Find mean user rating 
    sums = X.sum(axis=0)
    non_zeros = np.count_nonzero(X.toarray(), axis=0)
    means = sums.flatten() / (non_zeros+(non_zeros==0)) # Account for possible divide by 0

    # Adjust ratings by means
    X_transpose = X.transpose() 
    nz = X_transpose.nonzero()
    X_transpose = X_transpose.astype('float64')
    X_transpose[nz] -= means[0, nz[0]]

    # Assign mean adjusted data to self
    self.data = X_transpose.transpose()
    self.user_offset = means[0,:]
  
  def get_neighbors(self, X, verbose=False):
    """Get k closest neighbors by their cosine similarity"""

    # Find similarities
    similarities = cosine_similarity(X, self.data)
    index_arr = np.argsort(similarities, axis=1).flatten()[::-1] #reverse sorted array for descending order

    # Get K neighbors from similarities
    neighbors = index_arr[:self.K]
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