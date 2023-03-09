# pylint: skip-file
from sklearn.model_selection import train_test_split

def create_split(data, train_size, val_size):
  train, test = train_test_split(data, train_size=train_size)
  train, validation = train_test_split(train, test_size=val_size)
  return train, validation, test