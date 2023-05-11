import numpy as np
from daisy.model.MFRecommender import MF
from daisy.utils.sampler import BasicNegtiveSampler
from daisy.utils.dataset import get_dataloader, BasicDataset


class MFRec:
  def __init__(self, config, lr=0.01, reg_1=0.001, reg_2=0.001, factors=100): #elastic=0.1, alpha=1.0):
    config['lr'] = lr
    config['reg_1'] = reg_1
    config['reg_2'] = reg_2
    config['factors'] = factors
    self.config = config
    self.lr = lr
    self.reg_1 = reg_1
    self.reg_2 = reg_2
    self.factors = factors
    self.model = MF(config)

  def fit(self, X):
    ## May need to do preprocessing steps on train data
    sampler = BasicNegtiveSampler(X, self.config)
    train_samples = sampler.sampling()
    train_dataset = BasicDataset(train_samples)
    train_loader = get_dataloader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
    self.model.fit(train_loader)

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
  
  def set_params(self, lr=None, reg_1=None, reg_2=None, factors=None):
    if lr:
      self.config['lr'] = lr
      self.lr = lr
    if reg_1:
      self.config['reg_1'] = reg_1
      self.reg_1 = reg_1
    if reg_2:
      self.config['reg_2'] = reg_2
      self.reg_2 = reg_2
    if factors:
      self.config['factors'] = factors
      self.factors = factors
    self.model = MF(self.config)
    
  def __str__(self) -> str:
    return f'MF(lr={self.lr}, reg_1={self.reg_1}, reg_2={self.reg_2}, factors={self.factors})'