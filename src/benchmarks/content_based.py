from os import listdir
from os.path import isfile, join
import numpy as np

from src.dataloader import load_dataset
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing import preprocessor
# from src.algorithms import popularity, itemknn, slim, item2vec
from src.algorithms.cbknn import TFIDFKNN, Word2VecKNN
from src.utils import result_visualizer

from daisy.utils.config import init_config, init_seed, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, F1, Recall, Precision
from logging import getLogger

from sklearn.model_selection import train_test_split
config = init_config()
config['algo_name'] = 'itemknn'

''' init seed for reproducibility '''
init_seed(config['seed'], config['reproducibility'])

''' init logger '''
init_logger(config)
logger = getLogger()
logger.info(config)
config['logger'] = logger
config['topk'] = 10
config['maxk'] = 100
config['title_col'] = 'title'

'''Load and process datasets...'''
mypath = '../../data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]

for f in files[1:]: # TODO: CHANGE RANGE!!!
  print(f)
  # Load dataset
  if 'BX' in f:
    df = load_dataset.book_crossing(use_title=True)
    df = preprocessor.proces(df, k_filter=10)
  else:
    if 'rating_only' in f:
      df = load_dataset.amazon(f[:-16], use_title=True)
    elif 'steam' in f:
      df = load_dataset.steam(use_title=True)
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

  models = [TFIDFKNN(config),
            Word2VecKNN(config)]#,
            # item2vec.Item2VecRec(config)]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
  # metrics = ['n']
  results = {}
  results['dataset'] = f
  results['metrics'] = ['NDCG', 'Precision', 'Recall', 'F1']
  
  p = []
  for m in models:
    m.fit(train)
    ranks = m.rank(test)

    # print(ranks[:10])
    # ndcg_10 = NDCG(test_ur, ranks[:10], test_u)
    # ndcg_20 = NDCG(test_ur, ranks[:20], test_u)
    # ndcg_50 = NDCG(test_ur, ranks[:50], test_u)
    ndcg_full = NDCG(test_ur, ranks, test_u)
    precision = Precision(test_ur, ranks, test_u)
    recall = Recall(test_ur, ranks, test_u)
    f1 = F1(test_ur, ranks, test_u)
    results[str(m)] = [ndcg_full, precision, recall, f1]

  result_visualizer.build(results)
