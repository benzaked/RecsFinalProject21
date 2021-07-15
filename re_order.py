
import pandas as pd
import numpy as np
import ast
from collections import defaultdict

def cast_to_list_of_int(lst):
    if lst is not np.nan:
        return [int(p) for p in ast.literal_eval(lst)]


def get_duplicate_values_in_dict(dictionary):
    flipped = {}
    to_return = []
    for key, value in dictionary.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)
    flipped = dict(sorted(flipped.items(), key=lambda item: item[1]))
    for key, value in flipped.items():
        if len(value) > 1:
            to_return.extend(value)
    return to_return


def get_best_order(idx, base_order=['item_item', 'user_user']):
    recs = output.iloc[idx]['nais_predictions']
    i = 0
    best_order = {}
    for alg in base_order:
        item_list = output.iloc[idx][alg]
        if item_list:
            best_order[alg] = recs.index(item_list[0])
    best_order = dict(sorted(best_order.items(), key=lambda item: item[1]))
    return list(best_order.keys())

def get_best_order_by_avg(idx, base_order=['item_item', 'user_user']):
  recs = output.iloc[idx]['nais_predictions']
  base_order = [ 'content', 'item_item', 'user_user', 'popularity', 'general']
  if data_set == 'pinterest-20':
      base_order.remove('content')
  best_order = {}
  for alg in base_order:
      item_list = output.iloc[idx][alg]
      if item_list:
        l = []
        for item in item_list:
          l.append(recs.index(item))
        best_order[alg] = int(np.mean(l))
  best_order = dict(sorted(best_order.items(), key=lambda item: item[1]))
  return list(best_order.keys())


def get_items_by_ordered_key_list(idx, list_ordered):
    """
    :param idx:
    :param list_ordered:
    :return:
    """
    recs = output.iloc[idx]['nais_predictions']
    explained_recs = []
    for alg in list_ordered:
        if output.iloc[idx][alg]:
            explained_recs.extend(output.iloc[idx][alg])
    if len(recs) != len(explained_recs):
        print("you have a bug!!!!")
    return explained_recs

def get_display_order_list(idx):
    output_order = ['content']
    output_order.extend(get_best_order(idx))
    output_order.extend(get_best_order(idx,['popularity', 'general']))
    return get_items_by_ordered_key_list(idx, output_order)


def get_optimal_order_list(idx):
    """
    :param idx:
    :return: list_ordered
    """
    output_order = ['content', 'item_item', 'user_user', 'popularity', 'general']
    if data_set == 'pinterest-20':
        output_order.remove('content')
    output_order = get_best_order(idx, output_order)
    return get_items_by_ordered_key_list(idx, output_order)

def get_optimal_order_list_by_avg(idx):
    """
    :param idx:
    :return: list_ordered
    """
    output_order = [ 'content', 'item_item', 'user_user', 'popularity', 'general']
    if data_set == 'pinterest-20':
        output_order.remove('content')
    output_order = get_best_order_by_avg(idx, output_order)
    return get_items_by_ordered_key_list(idx, output_order)


def get_larger_size_order_list(idx):
    explain_algs = [ 'content', 'item_item', 'user_user', 'popularity', 'general']
    if data_set == 'pinterest-20':
        explain_algs.remove('content')
    explain_size_dict = {}
    for alg in explain_algs:
        if output.iloc[idx][alg]:
            explain_size_dict[alg] = len(output.iloc[idx][alg])
    explain_size_dict = dict(sorted(explain_size_dict.items(), reverse=True, key=lambda item: item[1]))
    duplicate_to_sort = get_duplicate_values_in_dict(explain_size_dict)
    explain_size_list = explain_size_dict.keys()
    explain_size_list = list(explain_size_list)
    # if we have the same size, we prefer the order that closest to the optimal
    if len(duplicate_to_sort) > 0:
        sorted_list = get_best_order(idx, duplicate_to_sort)
        opt = explain_size_list.copy()
        for i in range(len(duplicate_to_sort)):
            opt[explain_size_list.index(duplicate_to_sort[i])] = sorted_list[i]
        return get_items_by_ordered_key_list(idx, opt)
    return get_items_by_ordered_key_list(idx, explain_size_list)

data_set = 'pinterest-20'
output = pd.read_csv(f'predictions/{data_set}/output.csv')
output['nais_predictions'] = output['nais_predictions'].apply(lambda x : cast_to_list_of_int(x))
output['popularity'] = output['popularity'].apply(lambda x : cast_to_list_of_int(x))
output['general'] = output['general'].apply(lambda x : cast_to_list_of_int(x))
output['user_user'] = output['user_user'].apply(lambda x : cast_to_list_of_int(x))
output['item_item'] = output['item_item'].apply(lambda x : cast_to_list_of_int(x))
if data_set != 'pinterest-20':
    output['content'] = output['content'].apply(lambda x : cast_to_list_of_int(x))

output['ranked_item'] = output['ranked_item'].apply(lambda x : int(x))

re_ordered = pd.DataFrame()
for i in output.index:
    row = defaultdict(list)
    row['user_preferance'] = get_optimal_order_list(i)
    row['larger_size_order_list'] = get_larger_size_order_list(i)
    if data_set != 'pinterest-20':
        row['content_first'] = get_display_order_list(i)

    row['ranked_item'] = output.iloc[i]['ranked_item']
    row['user_preference_avg'] = get_optimal_order_list_by_avg(i)
    re_ordered = re_ordered.append(row,ignore_index=True)

re_ordered.to_csv(f'predictions/{data_set}/re_ordered_16.csv', index=False)
