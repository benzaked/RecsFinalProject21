import math
import numpy as np


def get_hit_ratio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    return 0


def get_NDCG(ranklist, gtItem):
    rank_list_np = np.array(ranklist)
    pos = np.where(rank_list_np == gtItem)
    if len(pos[0]) > 0:
        return math.log(2) / math.log(pos[0][0] + 2)
    return 0


def get_MRR(ranklist, gtItem):
    rank_list_np = np.array(ranklist)
    pos = np.where(rank_list_np == gtItem)
    if len(pos[0]) > 0:
        return 1 / (pos[0][0] + 1)
    return 0


def evaluate(ranklist, test_item):
    hr = get_hit_ratio(ranklist, test_item)
    ndcg = get_NDCG(ranklist, test_item)
    mrr = get_MRR(ranklist, test_item)
    return hr, ndcg, mrr
