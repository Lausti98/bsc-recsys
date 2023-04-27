from os import listdir
from os.path import isfile, join
import numpy as np

from src.dataloader import load_dataset
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing import preprocessor
from src.algorithms import popularity, itemknn, slim, item2vec
from src.model_tuning.model_tuning import grid_search
from src.utils import result_visualizer

from daisy.utils.config import init_config, init_seed, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, F1, Recall, Precision, HR
from logging import getLogger

from sklearn.model_selection import train_test_split
config = init_config()
config['algo_name'] = 'itemknn'

''' init seed for reproducibility '''
init_seed(config['seed'], config['reproducibility'])

''' init logger '''
config['state'] = 'warning' # silence info logs
init_logger(config)
logger = getLogger()
# logger.info(config)
config['logger'] = logger
config['topk'] = 100

'''Load and process datasets...'''
mypath = '../../data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]

for f in files:#[1:]:
  # Load dataset
  if 'BX' in f:
    df = load_dataset.book_crossing()
    df = preprocessor.proces(df, k_filter=10)
  else:
    if 'rating_only' in f:
      df = load_dataset.amazon(f[:-16])
    elif 'steam' in f:
      df = load_dataset.steam()
    df = preprocessor.proces(df)

  user_num = df['user'].nunique()
  item_num = df['item'].nunique()
  config['user_num'] = int(user_num)
  config['item_num'] = int(item_num)
  config['cand_num'] = int(item_num) # Use all items as candidate ranking
  # config['maxk'] = 100
  # config['topk'] = 50
  # config['shrink'] = 0.0
  # config['similarity'] = 'adjusted'
  # config['normalize'] = False

  train, test = train_test_split(df, train_size=0.7, random_state=1)

  # Ground truths
  test_ur = get_ur(test)
  total_train_ur = get_ur(train)
  config['train_ur'] = total_train_ur

  models = [popularity.Popularity(config),
            itemknn.ItemKNN(config, K=config['topk']),
            slim.SLiMRec(config)]#,
            # item2vec.Item2VecRec(config)]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
  # metrics = ['n']
  results = {}
  results['dataset'] = f
  results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
  param_grid = {'alpha': [1.0, 0.8, 0.6, 0.4, 0.2],
                'elastic': [0.1, 0.3, 0.5, 0.7, 0.9]}
  p = []
  for m in models:
    if isinstance(m, slim.SLiMRec):
      best_params = grid_search(train, m, config, param_grid)
      m.set_params(**best_params)
      print(f'grid search best params: {best_params}')
    m.fit(train)
    ranks = m.rank(test)

    ranks_10 = ranks[:,:10]
    ranks_20 = ranks[:,:20]
    ndcg_10 = NDCG(test_ur, ranks_10, test_u)
    ndcg_20 = NDCG(test_ur, ranks_20, test_u)
    # ndcg_50 = NDCG(test_ur, ranks[:50], test_u)
    ndcg_full = NDCG(test_ur, ranks, test_u)
    precision_10 = Precision(test_ur, ranks_10, test_u)
    precision_20 = Precision(test_ur, ranks_20, test_u)
    precision = Precision(test_ur, ranks, test_u)
    recall = Recall(test_ur, ranks, test_u)
    hr_10 = HR(test_ur, ranks_10, test_u)
    hr_20 = HR(test_ur, ranks_20, test_u)
    # f1 = F1(test_ur, ranks, test_u)
    results[str(m)] = [ndcg_10, ndcg_20, ndcg_full, precision_10, precision_20, precision, recall, hr_10, hr_20]

  result_visualizer.build(results)

