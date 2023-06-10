# pylint: skip-file
from src.dataloader.load_dataset import load_by_filepath
from src.preprocessing import preprocessor
from src.algorithms import popularity, itemknn, slim
from src.model_tuning.model_tuning import grid_search
from src.utils import result_visualizer
from src.utils.utils import get_csv_data_files

from daisy.utils.config import init_config, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, Recall, Precision, HR
from logging import getLogger

from sklearn.model_selection import train_test_split
config = init_config()
config['algo_name'] = 'itemknn'

config['state'] = 'warning'
init_logger(config)
logger = getLogger()
logger.info(config)
config['logger'] = logger
config['topk'] = 100

files = get_csv_data_files()

for f in files[4:]:
  # Load dataset
  df = load_by_filepath(f, use_title=False)
  if 'BX' in f:
    df = preprocessor.proces(df, k_filter=10)
    # config['topk'] = 50 # Result from grid search
  else:
    # if 'rating_only' in f:
    #   # config['topk'] = 30 # Result from gridsearch
    # elif 'steam' in f:
      #df = load_dataset.steam()
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

  models = [popularity.Popularity(config),
            itemknn.ItemKNN(config, K=config['topk']),
            slim.SLiMRec(config, elastic=0.1, alpha=0.2)]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
  
  results = {}
  results['dataset'] = f
  results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
  
  for m in models:
    if isinstance(m, slim.SLiMRec):
      param_grid = {'alpha': [0.1, 0.01, 0.001, 1.0, 0.8, 0.6, 0.4, 0.2],
              'elastic': [0.1, 0.3, 0.5, 0.7, 0.9]}
      best_params = grid_search(train, m, config, param_grid, verbose=True)
      m.set_params(**best_params)
      print(f'grid search best params: {best_params}')
    elif isinstance(m, itemknn.ItemKNN):
      params = {'K': [5, 10, 20, 30, 50, 70, 100]}
      best_params = grid_search(train, m, config, params, verbose=True)
      m.set_params(**best_params)
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

