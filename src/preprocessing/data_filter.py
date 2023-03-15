#pylint: skip-file
import pandas as pd

def k_core_filter(X, k):
  """Get data with over k interactions on items and users""" 
  user_interactions = X['user_id'].value_counts()
  X = X[X['user_id'].isin(user_interactions.index[user_interactions > k])]
  item_interactions = X['item_id'].value_counts()
  X = X[X['item_id'].isin(item_interactions.index[item_interactions > k])]
  return X

if __name__ == '__main__':
  df = pd.read_csv('../data/software_rating_only.csv', index_col=0, skiprows=1, names=['user_id', 'item_id', 'rating'])#.drop(columns=[0])
  print(df.head())
  print(df.shape)
  f_df = k_core_filter(df, 1)
  print(f_df.head())
  print(f_df.shape)