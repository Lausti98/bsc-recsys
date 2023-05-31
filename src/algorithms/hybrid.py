# pylint: skip-file
"""Module contains hybrid recommender algorithms"""
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class Switch:
  def __init__(self, config, cb_model, cf_model, cutoff=3, cutoff_on='user'):
    self.cb_model = cb_model
    self.cf_model = cf_model
    self.user_num = config['user_num']
    self.item_num = config['item_num']
    self.K = config['topk']
    self.cutoff = cutoff
    self.co = cutoff_on
    print(self.item_num)
    print(self.user_num)

  def fit(self, X):
    X = X.copy()

    if self.co == 'user':
      user_interaction_counts = X.groupby('user').count()['item']
      self.interaction_count = np.zeros(self.user_num)
      for usr, val in user_interaction_counts.items():
        self.interaction_count[usr] = val
    elif self.co == 'item':
      item_interaction_counts = X.groupby('item').count()['user']
      self.interaction_count = np.zeros(self.item_num)
      for item, val in item_interaction_counts.items():
        self.interaction_count[item] = val

    scaler = MinMaxScaler()
    self.scaler = scaler.fit(X['rating'].values.reshape(-1,1))

    self.X = X
    self.cb_model.fit(X)
    self.cf_model.fit(X)


  def rank(self, test_df):
    t_users = test_df['user'].unique()
    ranks = []
    if len(t_users) > 1:
      for user in t_users:
        ranks.append(self._ranker(user))
    else:
      ranks.append(self._ranker(user))
    
    return np.array(ranks)


  def set_params(self, cb_model=None, cf_model=None, cutoff=None, cutoff_on=None):
    if cb_model:
      self.cb_model = cb_model
    if cf_model:
      self.cf_model = cf_model
    if cutoff:
      self.cutoff = cutoff
    if cutoff_on:
      self.co = cutoff_on

  
  def _ranker(self, user):
    if self.co == 'user':
      if self.interaction_count[user] > self.cutoff:
        return self.cf_model.full_rank(user)
      else:
        return self.cb_model.full_rank(user)
    elif self.co == 'item':
      cb_preds = self.cb_model.predict(user)[0]
      cb_preds = self.scaler.transform(cb_preds.reshape(-1,1)).flatten()
      cf_preds = self.cf_model.predict(user)
      cf_preds = self.scaler.transform(cf_preds.reshape(-1,1)).flatten()
      scores = np.zeros(self.item_num)
      for i in range(self.item_num):
        if self.interaction_count[i] > self.cutoff:
          scores[i] = cf_preds[i]
        else:
          scores[i] = cb_preds[i]
      

      index_arr = np.argsort(scores).flatten()[::-1]
      ranks = index_arr[:self.K]

      return ranks


  def __str__(self) -> str:
    return f'Hybrid Switch(cb_model={str(self.cb_model)}, cf_model={str(self.cf_model)}, cutoff={self.cutoff}, cutoff_on={self.co})'



class Weighted:
  def __init__(self, config, cb_model, cf_model, alpha=0.1):
    self.cb_model = cb_model
    self.cf_model = cf_model
    self.alpha = alpha
    self.K = config['topk']
    pass

  def fit(self, X):
    X = X.copy()
    
    scaler = MinMaxScaler()
    self.scaler = scaler.fit(X['rating'].values.reshape(-1,1))

    self.items = X['item'].unique()

    self.X = X
    self.cb_model.fit(X)
    self.cf_model.fit(X)


  def rank(self, test_df):
    t_users = test_df['user'].unique()
    ranks = []
    if len(t_users) > 1:
      for user in t_users:
        ranks.append(self.full_rank(user))
    else:
      ranks.append(self.full_rank(user))
    
    return np.array(ranks)


  def full_rank(self, user):
    # Scores 
    cb_scores = self.cb_model.predict(user)[0]
    cb_scores = self.scaler.transform(cb_scores.reshape(-1,1)).flatten()

    cf_scores = self.cf_model.predict(user)
    cf_scores = self.scaler.transform(cf_scores.reshape(-1,1)).flatten()

    # combine scores
    scores = self.alpha*cb_scores + (1-self.alpha)*cf_scores

    scores = np.array(scores)
    index_arr = np.argsort(scores)[::-1]
    rank = index_arr[:self.K]

    return rank

  def set_params(self, cb_model=None, cf_model=None, alpha=None):
    if cb_model:
      self.cb_model = cb_model
    if cf_model:
      self.cf_model = cf_model
    if alpha:
      self.alpha = alpha


  def __str__(self) -> str:
    return f'Hybrid Weighted(cb_model={str(self.cb_model)}, cf_model={str(self.cf_model)}, alpha={self.alpha})'