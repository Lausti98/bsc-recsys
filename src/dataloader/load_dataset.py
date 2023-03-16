# pylint: skip-file
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_RELATIVE_PATH = '../../data/'
STEAM_DATASET_PATH = 'steam-200k.csv'

def steam(use_title=False):
  df = pd.read_csv(f'{DATA_RELATIVE_PATH}{STEAM_DATASET_PATH}')
  df = df[df.iloc[:, 2].astype('string') == 'play']
  df = df.iloc[:, [0,1,3]]
  df.columns.values[0:3] =['user_id', 'title', 'rating']
  encoder = LabelEncoder()
  df['user_id'] = encoder.fit_transform(df['user_id'])
  df['item_id'] = encoder.fit_transform(df['title'])
  if not use_title:
    df = df.drop(columns=['title'])
  
  return df

def amazon(category, use_title=False):
  df = pd.read_csv(f'{DATA_RELATIVE_PATH}{category}_rating_only.csv',
                  index_col=0,
                  skiprows=1,
                  names=['user_id', 'item_id', 'rating'])
  if use_title:
    meta_df = pd.read_json(
      f'{DATA_RELATIVE_PATH}raw/meta_{category}.json',
      lines=True)
    df['title'] = meta_df['title']

  return df


if __name__ == '__main__':
  a_df = amazon('amazon_fashion', use_title=True)
  print(a_df.head())