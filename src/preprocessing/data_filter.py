#pylint: skip-file
import pandas as pd

def k_core_filter(X, k):
  """Get data with over k interactions on items and users""" 
  usr_cnt = X.groupby(['user_id'], as_index=False)['item_id'].count()
  usr_cnt.rename(columns={'item_id': 'cnt_user'}, inplace=True)
  itm_cnt = X.groupby(['item_id'], as_index=False)['user_id'].count()
  itm_cnt.rename(columns={'user_id': 'cnt_item'}, inplace=True)
  X = X.merge(usr_cnt, on=['user_id']).merge(itm_cnt, on=['item_id'])
  
  X = X.query(f'cnt_item >= {k} and cnt_user >= {k}').reset_index(drop=True)
  X = X.drop(columns=['cnt_item', 'cnt_user'])
  return X

if __name__ == '__main__':
  df = pd.read_csv('../../data/software_rating_only.csv', index_col=0, skiprows=1, names=['user_id', 'item_id', 'rating', 'title'])#.drop(columns=[0])
  print(df.head())
  print(df.shape)
  f_df = k_core_filter(df, 3)
  print(f_df.head())
  print(f_df.shape)