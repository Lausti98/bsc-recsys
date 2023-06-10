# pylint: skip-file
from os import listdir
from os.path import isfile, join
import numpy as np

from src.dataloader.load_dataset import load_by_filepath
from src.preprocessing import preprocessor
from src.algorithms import itemknn, slim
from src.algorithms.cbknn import TFIDFKNN
from src.algorithms.hybrid import Switch, Weighted
from src.utils import result_visualizer
from src.model_tuning.model_tuning import grid_search

from daisy.utils.config import init_config, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, F1, Recall, Precision, HR
from logging import getLogger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
config = init_config()
config['algo_name'] = 'itemknn'


''' init logger '''
config['state'] = 'warning' # silence info logs
init_logger(config)
logger = getLogger()
logger.warn(config)
config['logger'] = logger
config['topk'] = 50
config['maxk'] = 500
config['title_col'] = 'title'

'''Load and process datasets...'''
mypath = '../../data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]

for f in files[2:]: # TODO: CHANGE RANGE!!!
  df = load_by_filepath(f, use_title=True)
  # Load dataset
  if 'BX' in f:
    df = preprocessor.proces(df, k_filter=10)
    alpha=0.2
    elastic=0.1
    config['topk']=50
  else:
    if 'rating_only' in f:
      alpha=0.01
      elastic=0.1
      config['topk']=30

    elif 'steam' in f:
      alpha=0.01
      elastic=0.1
      config['topk']=30
    df = preprocessor.proces(df)

  user_num = df['user'].nunique()
  item_num = df['item'].nunique()
  config['user_num'] = int(user_num)
  config['item_num'] = int(item_num)
  config['cand_num'] = int(item_num) # Use all items as candidate ranking

  train, test = train_test_split(df, train_size=0.7, random_state=1)

  # Ground truths
  test_ur = get_ur(test)
  total_train_ur = get_ur(train)
  config['train_ur'] = total_train_ur

  models = [
            Switch(
              config=config,
              cf_model=slim.SLiMRec(config, elastic=elastic, alpha=alpha),
              cb_model=TFIDFKNN(config),
              cutoff=3,
              cutoff_on='user'
            ),
            Weighted(
              config=config,
              cf_model=slim.SLiMRec(config, elastic=elastic, alpha=alpha),
              cb_model=TFIDFKNN(config),
              alpha=0.1
            )
          ]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
  # metrics = ['n']
  results = {}
  results['dataset'] = f
  results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
  
  p = []
  for m in models:
    if isinstance(m, Switch):
      param_grid={'cutoff': [0, 2, 5, 10, 15, 20, 2000],
                  'cutoff_on': ['item', 'user'],
                  'cf_model': [itemknn.ItemKNN(config, config['topk']),
                               slim.SLiMRec(config, elastic=elastic, alpha=alpha)]}
      best_params = grid_search(train, m, config, param_grid, verbose=True)
      m.set_params(**best_params)
      print(f'grid search best params: {best_params}')
    elif isinstance(m, Weighted):
      param_grid={'alpha': [0.0, 0.2, 0.4, 0.6, 0.8, 1],
                  'cf_model': [itemknn.ItemKNN(config, config['topk']),
                               slim.SLiMRec(config, elastic=elastic, alpha=alpha)]}
      best_params = grid_search(train, m, config, param_grid, verbose=True)
      m.set_params(**best_params)
      print(f'grid search best params: {best_params}')
    m.fit(train)
    ranks = m.rank(test)

    ranks_10 = ranks[:,:10]
    ranks_20 = ranks[:,:20]
    ndcg_10 = NDCG(test_ur, ranks_10, test_u)
    ndcg_20 = NDCG(test_ur, ranks_20, test_u)
    ndcg_full = NDCG(test_ur, ranks, test_u)
    precision_10 = Precision(test_ur, ranks_10, test_u)
    precision_20 = Precision(test_ur, ranks_20, test_u)
    precision = Precision(test_ur, ranks, test_u)
    recall = Recall(test_ur, ranks, test_u)
    hr_10 = HR(test_ur, ranks_10, test_u)
    hr_20 = HR(test_ur, ranks_20, test_u)
    results[str(m)] = [ndcg_10, ndcg_20, ndcg_full, precision_10, precision_20, precision, recall, hr_10, hr_20]

  result_visualizer.build(results)
