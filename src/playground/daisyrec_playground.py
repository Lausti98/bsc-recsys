from logging import getLogger
import numpy as np

# DAISY REC IMPORTS
# Models
from daisy.model.PopRecommender import MostPop
from daisy.model.KNNCFRecommender import ItemKNNCF, convert_df
from daisy.model.SLiMRecommender import SLiM
from daisy.model.Item2VecRecommender import Item2Vec

# Utils
from daisy.utils.config import init_config, init_seed, init_logger
from daisy.utils.utils import get_ur, build_candidates_set
from daisy.utils.dataset import get_dataloader, CandidatesDataset, BasicDataset
from daisy.utils.metrics import NDCG
from daisy.utils.sampler import SkipGramNegativeSampler

# SRC IMPORTS
from src.dataloader import load_dataset
from src.preprocessing.data_filter import k_core_filter
from src.preprocessing.data_split import create_split
from src.algorithms.itemknn import ItemKNN
from src.preprocessing.matrix import ItemInteractionMatrix

#SKLEARN IMPORTS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

config = init_config()
config['algo_name'] = 'itemknn'

''' init seed for reproducibility '''
init_seed(config['seed'], config['reproducibility'])

''' init logger '''
init_logger(config)
logger = getLogger()
logger.info(config)
config['logger'] = logger


'''Load data in dataframe'''
df = load_dataset.amazon('amazon_fashion')
# df = load_dataset.book_crossing()
df = df[df['rating'] > 0]
df = k_core_filter(df, 4)
user_num = df['user_id'].nunique()
item_num = df['item_id'].nunique()

encoder = LabelEncoder()
df['user_id'] = encoder.fit_transform(df['user_id'])
df['item_id'] = encoder.fit_transform(df['item_id'])
# print('checking user encoding')
# print(df['user_id'].min, df['user_id'].max)


df = df.rename(columns={'user_id': 'user', 'item_id': 'item'})
# X = ItemInteractionMatrix(df, item_col='item', user_col='user')

train, test = train_test_split(df, test_size=0.7, random_state=1)
inter_mat_train = convert_df(user_num, item_num, train).T
inter_mat_test = convert_df(user_num, item_num, test).T
print(user_num, item_num)
config['user_num'] = int(user_num)
config['item_num'] = int(item_num)
config['cand_num'] = int(item_num) # Use all items as candidate ranking
config['maxk'] = 100
config['topk'] = 50
config['shrink'] = 0.0
config['similarity'] = 'adjusted'
config['normalize'] = False

''' get ground truth '''
test_ur = get_ur(test)
# print(test_ur)
total_train_ur = get_ur(train)
config['train_ur'] = total_train_ur

# config['UID_NAME'] = 'user_id'
# config['IID_NAME'] = 'item_id'
# print(config)

'''Fit model on train data'''
baseline = MostPop(config)
baseline.fit(train)

# ITEM BASED CF
model = ItemKNNCF(config)
model.fit(train)

model2 = ItemKNN(K=config['topk'])
model2.fit(inter_mat_train)


config['elastic'] = 0.1
config['alpha'] = 1.0
slim = SLiM(config)
slim.fit(train)

config['lr'] = 0.001
config['epochs'] =  20
config['factors'] = 100
config['rho'] = 0.5
config['context_window'] = 2
item2vec = Item2Vec(config)
sampler = SkipGramNegativeSampler(train, config)
train_samples = sampler.sampling()
train_dataset = BasicDataset(train_samples)
train_loader = get_dataloader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
item2vec.fit(train_loader)

'''Explore predict returns'''
# print(train.iloc[0])
# for i in range(len(test)):
#   user = test.iloc[i]['user']
#   item = test.iloc[i]['item']
#   rating = test.iloc[i]['rating']
#   pred = model.predict(user, item)
#   # pred2 = model2.predict(inter_mat_test.getrow(item), user)
#   pred2 = model2.pred_from_mat(user, item)
#   if pred > 0.0 or pred2 > 0.0:
#     print(f'pred {i}: {pred:.2f}, {"pred2":<12} {i}: {pred2:.2f}, ground truth: {rating}')

''' build candidates set '''
logger.info('Start Calculating Metrics...')
test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
# print(test_u)
# print(test_ucands)
''' get predict result '''
logger.info('==========================')
logger.info('Generate recommend list...')
logger.info('==========================')
test_dataset = CandidatesDataset(test_ucands)
test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
# print(test.iloc[0])
# print(test_loader.dataset[0])
# ranks = model.rank(test_loader) # np.array (u, topk)
# print(ranks[0,:])
# test.apply(lambda row: print(row.item))
bl_ranks = []
f_ranks = []
f_ranks2 = []
slim_ranks = []
i2v_ranks = []
for _, row in test.iterrows():
  # print(row)
  # print(row['item'])
  b = baseline.full_rank(row['user'])
  n1 = model.full_rank(row['user'])
  n2 = model2.full_rank((row['user']))
  slim_ranks.append(slim.full_rank(row['user']))
  i2v_ranks.append(item2vec.full_rank(row['user']))
  bl_ranks.append(b)
  f_ranks.append(n1)
  f_ranks2.append(n2)
# print(ranks)
# print(test.iloc[0])
# print(np.array(f_ranks2)[0,:])
# print(np.array(f_ranks)[0,:])





'''Metrics'''
# met1 = NDCG(test_ur, ranks, test_u)
met1 = NDCG(test_ur, np.array(f_ranks), test_u)
met2 = NDCG(test_ur, np.array(f_ranks2), test_u)
base_met = NDCG(test_ur, np.array(bl_ranks), test_u)
slim_met = NDCG(test_ur, np.array(slim_ranks), test_u)
i2v_met = NDCG(test_ur, np.array(i2v_ranks), test_u)
print(f'NDCG MostPop ranks: {base_met}')
print(f'NDCG daisyRec ranks: {met1}')
print(f'NDCG IKNN ranks: {met2}')
print(f'NDCG SLiM ranks: {slim_met}')
print(f'NDCG Item2Vec ranks: {i2v_met}')


