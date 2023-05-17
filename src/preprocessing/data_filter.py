#pylint: skip-file
import pandas as pd

def k_core_filter(X, k):
  """Get data with over k interactions on items and users""" 
  tmp1 = X.groupby(['user_id'], as_index=False)['item_id'].count()
  tmp1.rename(columns={'item_id': 'cnt_user'}, inplace=True)
  tmp2 = X.groupby(['item_id'], as_index=False)['user_id'].count()
  tmp2.rename(columns={'user_id': 'cnt_item'}, inplace=True)
  X = X.merge(tmp1, on=['user_id']).merge(tmp2, on=['item_id'])
  
  X = X.query(f'cnt_item >= {k} and cnt_user >= {k}').reset_index(drop=True)
  return X

if __name__ == '__main__':
  df = pd.read_csv('../../data/software_rating_only.csv', index_col=0, skiprows=1, names=['user_id', 'item_id', 'rating', 'title'])#.drop(columns=[0])
  print(df.head())
  print(df.shape)
  f_df = k_core_filter(df, 3)
  print(f_df.head())
  print(f_df.shape)