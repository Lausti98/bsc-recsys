# pylint: skip-file
"""
  Module performs experiment where the datasets are filtered by users that
  have 1 interaction, 2 interactions and so forth while doing testing
"""
import os
from logging import getLogger

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from daisy.utils.config import init_config, init_seed, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.metrics import NDCG, F1, Recall, Precision, HR

from src.dataloader import load_dataset
from src.algorithms import itemknn, slim, matrix_factorization, cbknn, hybrid
from src.preprocessing import preprocessor
from src.utils import result_visualizer

from src.utils import utils

if __name__ == '__main__':
  data_files = utils.get_csv_data_files()

  config = init_config()
  config['algo_name'] = 'itemknn'

  ''' init seed for reproducibility '''
  init_seed(config['seed'], config['reproducibility'])

  ''' init logger '''
  config['state'] = 'warning' # silence info logs
  init_logger(config)
  logger = getLogger()
  logger.info(config)
  config['logger'] = logger
  config['topk'] = 20
  config['maxk'] = 150
  config['title_col'] = 'title'



  for f in data_files[0:]: # TODO: CHANGE to 0
    print(f"loading file {f}")
    df = load_dataset.load_by_filepath(f, use_title=True)
    if 'BX' in f:
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
      slim_alpha=0.01
      elastic=0.1

      config['topk']=30

      w_alpha = 0.1

      cf_modelname = 'slim'
      c = 3
      co = 'user'

    for x in range(3,15): # TODO: CHANGE TO 3
      f_df = preprocessor.proces(df, k_filter=x)
      print('dataset processed')

      user_num = f_df['user'].nunique()
      item_num = f_df['item'].nunique()
      config['user_num'] = int(user_num)
      config['item_num'] = int(item_num)
      config['cand_num'] = int(item_num) 

      # get train test split
      train, test = train_test_split(f_df, train_size=0.7, random_state=1)
      print('train test set splitted')

      total_train_ur = get_ur(train)
      config['train_ur'] = total_train_ur

      print('initializing model')
      # initialize model
      # model = itemknn.ItemKNN(config, K=config['topk'])
      model = slim.SLiMRec(config, elastic=0.1, alpha=slim_alpha)
      # model = cbknn.TFIDFKNN(config)
      # model = hybrid.Weighted(config,
      #                         cb_model=cbknn.TFIDFKNN(config),
      #                         cf_model=slim.SLiMRec(config, elastic=elastic, alpha=slim_alpha),
      #                         alpha=w_alpha
      #                         )
      # model = hybrid.Switch(config,
      #                         cb_model=cbknn.TFIDFKNN(config),
      #                         cf_model=slim.SLiMRec(config, elastic=elastic, alpha=slim_alpha) if cf_modelname == 'slim' else itemknn.ItemKNN(config, K=config['topk']),
      #                         cutoff=c,
      #                         cutoff_on=co
      #                         )
      # model = cbknn.Word2VecKNN(config, pretrained=True)
      print('fitting model')
      model.fit(train)
      print('model fitted')

      results = {}
      results['dataset'] = f
      results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
      print(f'ranking for num interactions: {x}')
      # filter the users in test set with x num
      # filtered_test = filter_dataset(df, x)
      if len(test) > 0:
        # Ground truths
        test_ur = get_ur(test)

        test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
        # predict rankings for the users
        ranks = model.rank(test)
        # report eval metrics

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
        results[str(model)] = [ndcg_10, ndcg_20, ndcg_full, precision_10, precision_20, precision, recall, hr_10, hr_20]

        result_visualizer.build(results)
      

    # plot_num_interactions(df, user=False, title=os.path.split(f)[1])

