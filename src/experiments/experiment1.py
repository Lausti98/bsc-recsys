#pylint: skip-file
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
from src.algorithms import itemknn, slim, matrix_factorization, cbknn
from src.preprocessing import preprocessor
from src.utils import result_visualizer

from src.utils import utils

def plot_num_interactions(df, user=True, title=''):

  if user:
    num_interactions = df.groupby('user')['item'].count()
  else:
    num_interactions = df.groupby('item')['user'].count()
  
  hist = plt.hist(num_interactions, bins=15, range=(1,15))
  plt.title(title)
  plt.show()

# Filter the dataset by function
def filter_num_interactions_df(full_df, test_df, num_interactions, user=True):
  """
    Returns a dataset with interactions where each user only
    has 'num_interactions'
  """
  if user:
    interaction_counts = full_df.groupby('user')['item'].count()
    keep_ids = interaction_counts[interaction_counts == num_interactions]
    mask = test_df['user'].isin(keep_ids.index)
  else:
    interaction_counts = full_df.groupby('item')['user'].count()
    keep_ids = interaction_counts[interaction_counts == num_interactions]
    #print(interaction_counts)
    #print(interaction_counts.unique())
    #print(keep_ids)
    mask = test_df['item'].isin(keep_ids.index)
    #print(mask)

  filtered_df = test_df.loc[mask]
  #print(filtered_df)
  
  return filtered_df


if __name__ == '__main__':
  #df = load_dataset.amazon('software', use_title=True)
  df = load_dataset.steam()
  print(df['user_id'].nunique())
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
  config['topk'] = 50
  config['maxk'] = 150
  config['title_col'] = 'title'



  for f in data_files[0:]:
    print(f"loading file {f}")
    df = load_dataset.load_by_filepath(f, use_title=True)
    if 'BX' in f:
      slim_alpha = 0.2
    elif 'fashion' in f:
      slim_alpha = 0.01
    elif 'prime' in f:
      slim_alpha = 0.01
    elif 'software' in f:
      slim_alpha = 0.001
    elif 'steam' in f:
      slim_alpha = 0.01

    
    df = preprocessor.proces(df, k_filter=4) # must filter 4 due to memory restrictions
    print('dataset processed')

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()
    config['user_num'] = int(user_num)
    config['item_num'] = int(item_num)
    config['cand_num'] = int(item_num) 

    # get train test split
    train, test = train_test_split(df, train_size=0.7, random_state=1)
    print('train test set splitted')

    total_train_ur = get_ur(train)
    config['train_ur'] = total_train_ur

    print('initializing model')
    # initialize model
    # model = itemknn.ItemKNN(config, K=config['topk'])
    model = slim.SLiMRec(config, elastic=0.1, alpha=slim_alpha)
    # model = cbknn.TFIDFKNN(config)
    # model = cbknn.Word2VecKNN(config, pretrained=True)
    print('fitting model')
    model.fit(train)
    print('model fitted')

    for x in range(2,15):
      results = {}
      results['dataset'] = f
      results['metrics'] = ['NDCG_10', 'NDCG_20', 'NDCG', 'Precision_10', 'Precision_20', 'Precision', 'Recall', 'Hit-Rate_10', 'Hit-Rate_20']
      print(f'ranking for num interactions: {x}')
      # filter the users in test set with x num
      filtered_test = filter_num_interactions_df(df, test, x, user=True)
      if len(filtered_test) > 0:
        # Ground truths
        test_ur = get_ur(filtered_test)

        test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
        # predict rankings for the users
        ranks = model.rank(filtered_test)
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

