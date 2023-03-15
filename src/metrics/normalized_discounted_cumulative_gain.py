#pylint: skip-file
"""Module computes normalized discounted cumulative gain"""
import numpy as np

def cumulative_discounted_gain(arr):
  nom = 2**arr-1
  denom = np.log2(np.arange(len(arr)) + 2)
  return np.sum(nom / denom)

def normalized_cumulative_discounted_gain(pred, true):
  dcg = cumulative_discounted_gain(pred)
  idcg = cumulative_discounted_gain(true)
  ncdg = dcg / idcg
  return ncdg

res_arr = np.array([2, 3, 5])
true_arr1 = np.array([2, 3, 5])
true_arr2 = np.array([3, 5, 2])
print(cumulative_discounted_gain(res_arr))
print(normalized_cumulative_discounted_gain(res_arr, true_arr1))
print(normalized_cumulative_discounted_gain(res_arr, true_arr2))

