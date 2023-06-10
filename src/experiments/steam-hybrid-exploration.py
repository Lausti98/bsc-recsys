#pylint: skip-file
from os import listdir
from os.path import isfile, join
import numpy as np

from src.dataloader import load_dataset
from src.preprocessing import preprocessor
from src.algorithms import popularity, itemknn, slim, item2vec
from src.algorithms.cbknn import TFIDFKNN, Word2VecKNN
from src.algorithms.hybrid import Weighted

from daisy.utils.config import init_config, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, HR
from logging import getLogger

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
config = init_config()
config['algo_name'] = 'itemknn'


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
  print(i)
  print(baselines[i])
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
            # itemknn.ItemKNN(config, K=50),
            # slim.SLiMRec(config, elastic=0.1, alpha=0.2),
            # TFIDFKNN(config),
            # Word2VecKNN(config),
            # Word2VecKNN(config, pretrained=True)
            Weighted(config,
                     cb_model=TFIDFKNN(config),
                     cf_model=slim.SLiMRec(config, elastic=0.1, alpha=m_alpha[i]),
                     alpha=0.0)
          ]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
  results = {}
  results['dataset'] = f
  results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
  
  p = []
  for m in models:
    alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    scores = np.zeros(len(alphas))
    for idx, alpha in enumerate(alphas):
      m.set_params(alpha=alpha)
      m.fit(train)
      ranks = m.rank(test)
      i_ranks = ranks[:,:10]
      i_ndcg = NDCG(test_ur, i_ranks, test_u)
      scores[idx] = i_ndcg
    
    plt.plot(alphas, scores, label='Weighted hybrid')
  
  plt.plot(alphas, [baselines[i] for _ in scores], linestyle='dotted', label='SLiM')
  plt.title(f)
  plt.xlabel('alpha')
  plt.ylabel('ndcg@10')
  plt.legend(loc ="lower right")
  plt.show()
    