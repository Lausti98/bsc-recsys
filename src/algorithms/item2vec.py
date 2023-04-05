import numpy as np
from daisy.model.Item2VecRecommender import Item2Vec
from daisy.utils.sampler import SkipGramNegativeSampler
from daisy.utils.dataset import get_dataloader, BasicDataset


class Item2VecRec:
  def __init__(self, config,
               lr=0.001,
               epochs=20,
               factors=100,
               rho=0.5,
               context_window=2):
    config['lr'] = lr
    config['epochs'] =  epochs
    config['factors'] = factors
    config['rho'] = rho
    config['context_window'] = context_window
    self.config = config
    self.model = Item2Vec(config)

  def fit(self, X):
    ## May need to do preprocessing steps on train data
    print('fitting....')
    print(X)
    sampler = SkipGramNegativeSampler(X, self.config)
    train_samples = sampler.sampling()
    train_dataset = BasicDataset(train_samples)
    print(train_dataset)
    train_loader = get_dataloader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
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
  
  def __str__(self) -> str:
    return f'Item2VecRec(lr={self.config["lr"]}, epochs={self.config["epochs"]}, factors={self.config["factors"]}, rho={self.config["rho"]})'