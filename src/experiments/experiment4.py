# pylint: skip-file
from os import listdir
from os.path import isfile, join
import numpy as np

from src.dataloader import load_dataset
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing import preprocessor
from src.algorithms import popularity, itemknn, slim, item2vec
from src.algorithms.cbknn import TFIDFKNN, Word2VecKNN
from src.utils import result_visualizer, utils

from daisy.utils.config import init_config, init_seed, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, F1, Recall, Precision, HR
from logging import getLogger

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
config = init_config()
config['algo_name'] = 'itemknn'

''' init seed for reproducibility '''
init_seed(config['seed'], config['reproducibility'])

''' init logger '''
config['state'] = 'warning' # silence info logs
init_logger(config)
logger = getLogger()
logger.warn(config)
config['logger'] = logger
config['topk'] = 50
config['maxk'] = 100
config['title_col'] = 'title'

'''Load and process datasets...'''
mypath = '../../data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]

for f in files[0:]: # TODO: CHANGE RANGE!!!
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
  config['maxk'] = 150
  config['topk'] = 50
  config['shrink'] = 0.0
  config['similarity'] = 'adjusted'
  config['normalize'] = False

  train, test = train_test_split(df, train_size=0.7, random_state=1)

  # Ground truths
  test_ur = get_ur(test)
  total_train_ur = get_ur(train)
  config['train_ur'] = total_train_ur

  models = [
            itemknn.ItemKNN(config, K=50),
            slim.SLiMRec(config, elastic=0.1, alpha=0.2),
            TFIDFKNN(config),
            # Word2VecKNN(config),
            # Word2VecKNN(config, pretrained=True)
          ]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
  # metrics = ['n']
  results = {}
  results['dataset'] = f
  results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
  
  p = []
  for m in models:
    m.fit(train)
    ranks = m.rank(test)

    scores = np.zeros(50)
    for i in range(1,50):
      i_ranks = ranks[:,:i]
      i_ndcg = NDCG(test_ur, i_ranks, test_u)
      scores[i] = i_ndcg
    
    plt.plot(scores, label=str(m))
  plt.title(f)
  plt.xlabel('k')
  plt.ylabel('ndcg@k')
  plt.legend(loc ="lower right")
  plt.show()
    


  #   ranks_10 = ranks[:,:10]
  #   ranks_20 = ranks[:,:20]
  #   ndcg_10 = NDCG(test_ur, ranks_10, test_u)
  #   ndcg_20 = NDCG(test_ur, ranks_20, test_u)
  #   ndcg_full = NDCG(test_ur, ranks, test_u)
  #   precision_10 = Precision(test_ur, ranks_10, test_u)
  #   precision_20 = Precision(test_ur, ranks_20, test_u)
  #   precision = Precision(test_ur, ranks, test_u)
  #   recall = Recall(test_ur, ranks, test_u)
  #   hr_10 = HR(test_ur, ranks_10, test_u)
  #   hr_20 = HR(test_ur, ranks_20, test_u)
  #   results[str(m)] = [ndcg_10, ndcg_20, ndcg_full, precision_10, precision_20, precision, recall, hr_10, hr_20]

  # result_visualizer.build(results)