from random import shuffle
import pandas as pd


def get_user_group(user):
    clusters = pd.DataFrame(pd.read_csv('server_data/clusters.csv', index_col=0)['cluster'])

    if user in clusters.index:
        return clusters.loc[user].item()
    else:
        last_cluster = clusters.tail(1)['cluster'].item()
        clusters.append(pd.Series({'cluster': not last_cluster}, name=user)).to_csv('server_data/clusters.csv')
        return int(not last_cluster)




def shuffle_dict(d):
    keys = list(d.keys())
    shuffle(keys)
    shuffle_d = {}
    for k in keys:
        shuffle_d[k] = d[k]
    return shuffle_d
