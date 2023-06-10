# pylint: skip-file
from os import listdir
from os.path import isfile, join
import numpy as np

from src.dataloader import load_dataset
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing import preprocessor
from src.algorithms import popularity, itemknn, slim, item2vec, hybrid
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
  df = load_dataset.load_by_filepath(f, use_title=True)
  if 'BX' in f:
    df = preprocessor.proces(df, k_filter=10)
    # SLiM params
    slim_alpha=0.2
    elastic=0.1
    # ItemKNN params
    config['topk']=50
    # Weighted params
    w_alpha = 0.1
    # Switching params
    cf_modelname = 'slim'
    c = 5
    co = 'item'
  elif 'rating_only' in f:
    df = preprocessor.proces(df)
    # SLiM params
    slim_alpha=0.01
    elastic=0.1
    # ItemKNN params
    config['topk']=30

    if 'fashion' in f:
      #weighted params
      w_alpha = 0.2
      # Switching params
      cf_modelname = 'slim'
      c = 15
      co = 'item'
    else:
      w_alpha = 0.6
      # Switching params
      c = 20
      co = 'item'
      if 'prime' in f:
        cf_modelname = 'itemknn'
      cf_modelname = 'slim'
  elif 'steam' in f:
    df = preprocessor.proces(df)
    slim_alpha=0.01
    elastic=0.1

    config['topk']=30

    w_alpha = 0.1

    cf_modelname = 'slim'
    c = 3
    co = 'user'

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
            slim.SLiMRec(config, elastic=0.1, alpha=slim_alpha),
            TFIDFKNN(config),
            
            hybrid.Weighted(config,
                cb_model=TFIDFKNN(config),
                cf_model=slim.SLiMRec(config, elastic=elastic, alpha=slim_alpha),
                alpha=w_alpha
                ),
            hybrid.Switch(config,
                cb_model=TFIDFKNN(config),
                cf_model=slim.SLiMRec(config, elastic=elastic, alpha=slim_alpha) if cf_modelname == 'slim' else itemknn.ItemKNN(config, K=config['topk']),
                cutoff=c,
                cutoff_on=co
                )
            # Word2VecKNN(config),
            # Word2VecKNN(config, pretrained=True)
          ]
  
  test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)

  p = []
  for m in models:
    m.fit(train)
    ranks = m.rank(test)

    scores = np.zeros(20)
    for i in range(1,20):
      i_ranks = ranks[:,:i]
      i_ndcg = NDCG(test_ur, i_ranks, test_u)
      scores[i] = i_ndcg
    
    if isinstance(m, hybrid.Switch):
      plt.plot(scores, label='Hybrid Switch')
    elif isinstance(m, hybrid.Weighted):
      plt.plot(scores, label='Hybrid Weighted')
    else:
      plt.plot(scores, label=str(m))
  plt.title(f)
  plt.xlabel('k')
  plt.ylabel('ndcg@k')
  plt.legend(loc ="lower right")
  plt.show()
