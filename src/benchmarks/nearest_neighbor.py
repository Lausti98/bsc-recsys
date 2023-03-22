# pylint: skip-file
# import pandas as pd
import numpy as np
from scipy.sparse import hstack
from src.dataloader import load_dataset
from src.algorithms.itemknn import ItemKNN
from src.algorithms.popularity import Popularity
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing.data_split import create_split
from src.preprocessing.matrix import ItemInteractionMatrix
from src.feature_extraction.vectorize import get_tf_idf
from src.metrics.normalized_discounted_cumulative_gain import normalized_cumulative_discounted_gain

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
# data = load_dataset.amazon('amazon_fashion')
# data = load_dataset.steam()
data = load_dataset.book_crossing()
s_train, s_test = prepare(data)
iKNN = ItemKNN(K=100)
iKNN.fit(s_train)

# Baseline model
b_model = Popularity(K=100)
b_model = b_model.fit(data)


# Prepare data with ratings in column for testing.
nz = s_test.nonzero()
print(nz)

# _, neigh = iKNN.get_neighbors(s_test.getrow(0), verbose=True)
preds = []
true = []
for row, col in zip(nz[0], nz[1]):
  pred = iKNN.predict(s_test.getrow(row), col)
  preds.append(pred)
  true.append(s_test[row,col])

gains = normalized_cumulative_discounted_gain(np.array(preds), np.array(true))
print(f'gain: {gains}')



