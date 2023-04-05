from src.preprocessing.data_filter import k_core_filter

from sklearn.preprocessing import LabelEncoder


def proces(df, k_filter=4):
  df = df[df['rating'] > 0] #Remove '0' ratings
  df = k_core_filter(df, k_filter)

  encoder = LabelEncoder()
  df['user_id'] = encoder.fit_transform(df['user_id'])
  df['item_id'] = encoder.fit_transform(df['item_id'])


  df = df.rename(columns={'user_id': 'user', 'item_id': 'item'})

  return df