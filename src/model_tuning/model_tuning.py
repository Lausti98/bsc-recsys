from sklearn.model_selection import StratifiedKFold
from itertools import product
import pandas as pd
import numpy as np

from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, F1, Recall, Precision, HR

from src.algorithms.slim import SLiMRec


def grid_search(data, clf, config, param_grid, k_folds=5):
  cv_stf = StratifiedKFold(n_splits=k_folds)
  best_score = 0
  best_params = None
  for values in product(*param_grid.values()):
    dct = dict(zip(param_grid.keys(), values))
    clf.set_params(**dct)
    scores = []
    for train_index, test_index in cv_stf.split(data.drop(columns='rating'), data['rating']):
      train_df = data.iloc[train_index]#, 
      test_df = data.iloc[test_index]
      test_ur = get_ur(test_df)
      total_train_ur = get_ur(train_df)
      config['train_ur'] = total_train_ur
      test_u, _ = build_candidates_set(test_ur, total_train_ur, config)
      clf.fit(train_df)
      ranks = clf.rank(test_df)
      ranks_10 = ranks[:,:10]
      ndcg_10 = NDCG(test_ur, ranks_10, test_u)
      scores.append(ndcg_10)
    
    mean_score = np.array(scores).mean()
    if mean_score > best_score:
      best_score = mean_score
      best_params = dct
  
  return best_params


if __name__ == '__main__':
  from src.dataloader import load_dataset
  df = load_dataset.amazon('amazon_fashion')
  print(df)
  param_grid = {'alpha': [1.0, 0.8, 0.6, 0.4, 0.2],
                'elastic': [0.1, 0.3, 0.5, 0.7, 0.9]}
  train_X = df.drop(columns='rating')
  train_y = df['rating']
  grid_search(train_X, train_y, 'classifier', param_grid)
  
  for values in product(*param_grid.values()):
    print(zip(param_grid.keys(), values))
    print(dict(zip(param_grid.keys(), values)))