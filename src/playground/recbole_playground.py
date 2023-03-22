#pylint: skip-file
from recbole.model.general_recommender import ItemKNN
from src.dataloader import load_dataset
# from src.benchmarks.nearest_neighbor import prepare

df = load_dataset.steam()
# csr = prepare(df)

# print(csr)

model = ItemKNN(config={'k': 10, 'USER_ID_FIELD': 'user_id', 'ITEM_ID_FIELD': 'item_id', 'NEG_PREFIX': ''}, dataset=df)
print(model)