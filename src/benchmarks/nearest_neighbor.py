# pylint: skip-file
import pandas as pd
import numpy as np
from src.algorithms.itemknn import ItemKNN
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing.data_split import create_split
from src.preprocessing.matrix import ItemInteractionMatrix

DATA_RELATIVE_PATH = '../../data/'

def dataloader(fp, index_col=None):
  df = pd.read_csv(f'{DATA_RELATIVE_PATH}{fp}',
                  index_col=index_col if index_col else None,
                  skiprows=1,
                  names=['user_id', 'item_id', 'rating'])

  df = k_core_filter(df, 4)
  X = ItemInteractionMatrix(df)

  train, _, test = create_split(X, 0.7)
  return train, test

a_train, a_test = dataloader('amazon_fashion_rating_only.csv', index_col=0)

iKNN = ItemKNN()
iKNN.fit(a_train)

_, neigh = iKNN.get_neighbors(a_test.getrow(0), verbose=True)






