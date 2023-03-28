#pylint: skip-file
from recbole.model.general_recommender import ItemKNN, Pop
from recbole.data import Interaction
from src.dataloader import load_dataset
# from src.benchmarks.nearest_neighbor import prepare


df = load_dataset.steam()
# csr = prepare(df)

interaction = Interaction(df.sort_values(['user_id', 'rating']))
# print(csr)

model = Pop(dataset=interaction)
print(model)