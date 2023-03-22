# pylint: skip-file
import pandas as pd

from surprise import Dataset, NormalPredictor, KNNWithMeans , Reader
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import cross_validate, PredefinedKFold

from src.dataloader import load_dataset
from src.preprocessing.data_split import create_split



# df = pd.DataFrame(ratings_dict)
df = load_dataset.amazon('software')
print(df.head())
train, _, test = create_split(df, 0.7)

train.to_csv('train.csv', index=False, header=False)
test.to_csv('test.csv', index=False, header=False)
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5), sep=',')

# The columns must correspond to user id, item id and ratings (in that order).
# dsauto = DatasetAutoFolds(reader=reader, df=train)
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
print(data)

# pdkf = PredefinedKFold()

# trainset, testset = pdkf.split(data)

# knn = KNNWithMeans(K=100)
# knn = knn.fit(trainset)
# print(knn)
# We can now use this dataset as we please, e.g. calling cross_validate
res = cross_validate(KNNWithMeans(), data, cv=10)
# print(res)