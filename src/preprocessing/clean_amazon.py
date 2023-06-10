import pandas as pd
from sklearn import preprocessing

fname = 'Prime_Pantry.json'
fpath = f'../../data/raw/{fname}'
meta_fpath = f'../../data/raw/meta_{fname}'

# Load and proces data
df = pd.read_json(fpath, lines=True)
meta = pd.read_json(meta_fpath, lines=True)
le = preprocessing.LabelEncoder()
# use only verified reviews
df = df[df['verified'] == True]

# Merge title for each product
df = df.merge(meta[['asin', 'title']], on='asin', how='inner')

df['product_id'] = le.fit_transform(df['asin'])
df['user_id'] = le.fit_transform(df['reviewerID'])

# Remove unneeded columns
df = df[['user_id', 'product_id', 'overall', 'title']]
df.to_csv(f'../../data/{fname.split(sep=".")[0].lower()}_rating_only.csv')