import pandas as pd
from sklearn import preprocessing

fname = 'Prime_Pantry.json'
fpath = f'data/raw/{fname}'

# Load and proces data
df = pd.read_json(fpath, lines=True)
le = preprocessing.LabelEncoder()
# use only verified reviews
df = df[df['verified'] == True]
df['product_id'] = le.fit_transform(df['asin'])
df['user_id'] = le.fit_transform(df['reviewerID'])
print(df.dtypes)

# Remove unneeded columns
df = df[['user_id', 'product_id', 'overall']]
df.to_csv(f'data/{fname.split(sep=".")[0].lower()}_rating_only.csv')