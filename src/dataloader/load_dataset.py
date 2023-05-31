# pylint: skip-file
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import utils

DATA_RELATIVE_PATH = f'{utils.get_base_path()}data/'
STEAM_DATASET_PATH = 'steam-200k.csv'

def steam(use_title=False):
  df = pd.read_csv(f'{DATA_RELATIVE_PATH}{STEAM_DATASET_PATH}')
  df = df[df.iloc[:, 2].astype('string') == 'purchase']
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
                  names=['user_id', 'item_id', 'rating', 'title'])
  
  if not use_title:
    df = df.drop(columns='title')

  return df

def book_crossing(use_title=False):
  df = pd.read_csv(f'{DATA_RELATIVE_PATH}BX-Book-Ratings.csv',
                  on_bad_lines='skip', 
                  encoding_errors='replace',
                  sep=';',
                  skiprows=1,
                  names=['user_id', 'item_id', 'rating'])
  df['rating'] = df['rating'].astype('int8')
  if use_title:
    meta_df = pd.read_csv(f'{DATA_RELATIVE_PATH}raw/BX-Books.csv',
                          on_bad_lines='skip', 
                          encoding_errors='replace',
                          sep=';')
    meta_df.rename(columns={'ISBN': 'item_id', 'Book-Title': 'title'}, inplace=True)
    df = df.merge(meta_df[['item_id', 'title']], on='item_id', how='inner')
    # df['title'] = meta_df['Book-Title']

  return df


def load_by_filepath(f_path, use_title=False):
  if 'BX' in f_path:
    df = book_crossing(use_title=use_title)
  else:
    if 'rating_only' in f_path:
      df = amazon(f_path[:-16], use_title=use_title)
    elif 'steam' in f_path:
      df = steam(use_title=use_title)
  
  return df
    

if __name__ == '__main__':
  a_df = amazon('amazon_fashion', use_title=True)
  print(a_df.head())