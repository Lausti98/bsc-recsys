# pylint: skip-file
import numpy as np
from daisy.model.PopRecommender import MostPop

class Popularity:
  def __init__(self, config):
    self.K = config['topk']
    self.model = MostPop(config)

  def fit(self, X):
    #TODO: Do checks on the input DF/Interactionmatrix
    
    # self.data = X
    # item_interaction_counts = X.groupby(["item_id"])["item_id"].count().nlargest(self.K) 
    # self.topk = item_interaction_counts / len(X)
    # print(self.topk)
    self.model.fit(X)
  
  def predict(self, X, user):
    #TODO: Implement actual predictions 

    return self.topk
  
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