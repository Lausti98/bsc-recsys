"""Module visualizes the recommendation results"""

def build(result_dct):
  dataset = result_dct.pop('dataset')
  print(f'----- {dataset} results -----')
  metrics = result_dct.pop('metrics')
  print(f'{"metrics":<32}: {"    ".join(metrics)}')
  for key in result_dct.keys():
    formatted_list = ["%.4f"%item for item in result_dct[key]]
    print(f'{key:<32}: {formatted_list}')
