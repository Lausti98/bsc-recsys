import numpy as np
from daisy.model.SLiMRecommender import SLiM


class SLiMRec:
  def __init__(self, config, elastic=0.1, alpha=1.0):
    config['elastic'] = elastic
    config['alpha'] = alpha
    self.elastic = elastic
    self.alpha = alpha
    self.model = SLiM(config)

  def fit(self, X):
    ## May need to do preprocessing steps on train data
    self.model.fit(X)

  def predict():
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
  
  def set_params(self, alpha=None, elastic=None):
    if alpha:
      self.alpha=alpha
    if elastic:
      self.elastic=elastic
    
  def __str__(self) -> str:
    return f'SLiMRec(elastic={self.elastic}, alpha={self.alpha})'