# pylint: skip-file
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix

class Popularity:
  def __init__(self, K=10):
    self.K = K

  def fit(self, X):
    #TODO: Do checks on the input DF/Interactionmatrix
    
    self.data = X
    item_interaction_counts = X.groupby(["item_id"])["item_id"].count().nlargest(self.K) 
    self.topk = item_interaction_counts / len(X)
    print(self.topk)
  
  def predict(self, X, user):
    #TODO: Implement actual predictions 

    return self.topk