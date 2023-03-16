# pylint: skip-file
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from src.dataloader import load_dataset
from src.algorithms.itemknn import ItemKNN
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing.data_split import create_split
from src.preprocessing.matrix import ItemInteractionMatrix
from src.feature_extraction.vectorize import get_tf_idf

DATA_RELATIVE_PATH = '../../data/'

def prepare(df):
  df = k_core_filter(df, 4)
  X = ItemInteractionMatrix(df)
  if 'title' in df.columns:
    tmp_df = df.drop_duplicates(subset='item_id')
    title_csr = get_tf_idf(tmp_df['title'])
    X = hstack([X, title_csr])

  train, _, test = create_split(X, 0.7)

  return train, test

# a_train, a_test = dataloader('amazon_fashion_rating_only.csv', index_col=0)
data = load_dataset.steam(use_title=True)
s_train, s_test = prepare(data)
iKNN = ItemKNN()
iKNN.fit(s_train)

_, neigh = iKNN.get_neighbors(s_test.getrow(0), verbose=True)






