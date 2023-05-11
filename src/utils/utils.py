"""Module contains various utility functions"""
import os
from os import listdir
from os.path import isfile, join

def get_base_path():
  """Get the base path of project"""

  cur_dir = os.getcwd()
  src_path = cur_dir.rfind('src')

  if src_path > 0:
    base_path = cur_dir[0:src_path]
  else:
    # Already in base directory
    base_path = f'{cur_dir}/'

  return base_path

def get_csv_data_files():
  mypath = f'{get_base_path()}data/'
  files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.csv')]
  return files