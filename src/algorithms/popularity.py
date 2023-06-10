# pylint: skip-file
import numpy as np
from daisy.model.PopRecommender import MostPop

class Popularity:
  def __init__(self, config):
    self.K = config['topk']
    self.model = MostPop(config)

  def fit(self, X):
    self.model.fit(X)
  
  def predict(self, X, user):
    pass
  
  def rank(self, test_df):
    t_users = test_df['user'].unique()
    ranks = []
    if len(t_users) > 1:
      for user in t_users:
        ranks.append(self.model.full_rank(user))
    else:
      ranks.append(self.model.full_rank(user))
    
    return np.array(ranks)
  
  def __str__(self) -> str:
    return f'Popularity(K={self.K})'