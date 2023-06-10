#pylint: skip-file
from os import listdir
from os.path import isfile, join
import numpy as np

from src.dataloader.load_dataset import load_by_filepath
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing import preprocessor
from src.algorithms import popularity, itemknn, slim, item2vec
from src.algorithms.cbknn import TFIDFKNN, Word2VecKNN
from src.algorithms.hybrid import Weighted
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
baselines=[0.1202, 0.3128, 0.0952, 0.2496, 0.5634]
m_alpha = [0.2, 0.01, 0.01, 0.001, 0.01]

for i, f in enumerate(files):
  # Load dataset
  df = load_by_filepath(f, use_title=True)
  if 'BX' in f:
    df = preprocessor.proces(df, k_filter=10)
  else:
    df = preprocessor.proces(df)

  user_num = df['user'].nunique()
  item_num = df['item'].nunique()
  config['user_num'] = int(user_num)
  config['item_num'] = int(item_num)
  config['cand_num'] = int(item_num) # Use all items as candidate ranking
  config['maxk'] = 150
  config['topk'] = 30
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
            slim.SLiMRec(config, elastic=0.1, alpha=m_alpha[i]),
            # TFIDFKNN(config),
            # Word2VecKNN(config),
            # Word2VecKNN(config, pretrained=True)
            # Weighted(config,
            #          cb_model=TFIDFKNN(config),
            #          cf_model=slim.SLiMRec(config, elastic=0.1, alpha=m_alpha[i]),
            #          alpha=0.0)
          ]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
  # metrics = ['n']
  results = {}
  results['dataset'] = f
  results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
  
  p = []
  for m in models:
    m.fit(train)
    sim_num = np.zeros(user_num)
    for u in range(user_num):
      sim_num[u] = m._get_num_similar(u)
    b = np.where(sim_num < 50)

    plt.hist(sim_num[b], bins=10)
    plt.title(f'Number of similar items {str(m)} - {f}')
    plt.ylabel('count users')
    plt.xlabel('count items (sim > 0)')
    plt.show()
  
  

    