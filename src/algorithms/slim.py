# pylint: skip-file
import numpy as np
from daisy.model.SLiMRecommender import SLiM

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.simplefilter('ignore')

class SLiMRec:
  def __init__(self, config, elastic=0.1, alpha=1.0, scale=False):
    config['elastic'] = elastic
    config['alpha'] = alpha
    self.config = config
    self.elastic = elastic
    self.alpha = alpha
    self.model = SLiM(config)
    self.scale = scale

  def fit(self, X):
    ## May need to do preprocessing steps on train data
    self.X = X.copy()
    if self.scale:
      scaler = MinMaxScaler()
      self.X['rating'] = scaler.fit_transform(
        self.X['rating'].values.reshape(-1,1))

    self.model.fit(self.X)

  def predict(self, user, item=None):
    if item: 
      return self.model.predict(user, item)
    else:
      return self.model.A_tilde[user, :].A.squeeze()

  def rank(self, test_df):
    t_users = test_df['user'].unique()
    ranks = []
    if len(t_users) > 1:
      for user in t_users:
        ranks.append(self.model.full_rank(user))
    else:
      ranks.append(self.model.full_rank(test_df['user']))
    
    return np.array(ranks)
  
  def full_rank(self, user):
    return self.model.full_rank(user)

  def set_params(self, alpha=None, elastic=None):
    if alpha:
      self.alpha = alpha
      self.config['alpha'] = alpha
    if elastic:
      self.elastic = elastic
      self.config['elastic'] = elastic
    self.model = SLiM(self.config)
    
  def __str__(self) -> str:
    return f'SLiMRec(elastic={self.elastic}, alpha={self.alpha})'