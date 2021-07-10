from scipy import sparse
import pandas as pd
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import re
import ast
from sklearn import preprocessing
from dataset import DataSet



def clean_predictions( df, alg='not nais'):
  if alg == 'nais':
    df['predicted_list'] = df['predicted_list'].apply(lambda x: x.replace(',',''))
  df['predicted_list'] = df['predicted_list'].apply(lambda x: re.sub(r'[\[\]]', '', x).split())
  # df['predicted_score'] = df['predicted_score'].apply(lambda x: re.sub(r'[\[\]]', '', x).split())
  df['ranked_item'] = df['ranked_item'].apply(lambda x: str(x))
  return df


class RecommendationsAlgorithms:
    def __init__(self, data_item ,test_negative, dataset='ml-1m'):
        self.data_item = data_item
        self.item_mean, self.sim_matrix = self.get_item_item_params()
        self.projects_popularity_scores = data_item.astype(bool).sum(axis=0)
        self.model_knn = NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='brute', leaf_size=30, metric='cosine')
        self.model_knn.fit(self.data_item)
        self.test_negative = test_negative
        self.item_item_predictions = []
        self.user_user_predictions = []
        self.content_predictions = []
        self.nais_predictions = []
        self.dataset = dataset

    def read_predictions(self, dataset):
        self.nais_predictions = pd.read_csv(f'predictions/{dataset}/nais.csv')
        self.item_item_predictions = pd.read_csv(f'predictions/{dataset}/item_item.csv')
        self.user_user_predictions = pd.read_csv(f'predictions/{dataset}/user_user.csv')
        if dataset == 'ml-1m':
            self.content_predictions = pd.read_csv(f'predictions/{dataset}/content.csv')
            self.content_predictions = clean_predictions(self.content_predictions)

        self.item_item_predictions = clean_predictions(self.item_item_predictions)
        self.user_user_predictions = clean_predictions(self.user_user_predictions)
        self.nais_predictions = clean_predictions(self.nais_predictions, 'nais')

    def get_content_sim(movies_df):
      similarities = cosine_similarity(movies_df.drop(['movie_id', 'movie_title', 'movie_genre', 'title', 'year','movie_true_id'],1))
      sim = pd.DataFrame(similarities, movies_df.index, movies_df.index)
      sim.columns = [int(i) for i in sim.columns]
      sim.index = [int(i) for i in sim.index]
      sim = pd.DataFrame(preprocessing.StandardScaler().fit(sim).transform(sim),
                          columns=sim.columns, index=sim.index)
      return sim

    def find_k_similar_users(self, user_index, k=20):
        distances, indices = self.model_knn.kneighbors(
            np.array(self.data_item.iloc[user_index, :]).reshape(1, -1), n_neighbors=k + 1)
        similarities = 1 - distances.flatten()
        return pd.Series(similarities, indices[0])

    def get_recommendations_user_user(self, user_index, known_user_projects, test_items, k):
        similar_users = self.find_k_similar_users(user_index)
        if user_index in similar_users.index:
            similar_users = similar_users.drop(user_index, 0)
        similar_projects = [self.get_user_projects(user, self.data_item) for user in similar_users.index]
        similar_projects = list(set(chain(*similar_projects)))  # get all the projects from the dataframe
        projects_scores = dict.fromkeys(similar_projects, 0)
        for project in known_user_projects:
            if project in projects_scores:
                del projects_scores[project]
        # removing not relevant projects
        d = dict(projects_scores)
        project_naibors = list(d.keys())
        t = test_items
        for k in project_naibors:
            if k not in t:
                del d[k]
        projects_scores = d
        t = np.array(similar_users).dot(self.data_item[projects_scores.keys()].loc[similar_users.index])
        projects_scores_to_fit = np.array(t).reshape(-1, 1)
        scaled_scores = preprocessing.StandardScaler().fit(projects_scores_to_fit).transform(projects_scores_to_fit)
        projects_scores = pd.DataFrame(scaled_scores, index=pd.Series(projects_scores).index)[0]
        return pd.Series(projects_scores).astype(float).nlargest(20)

    def get_item_item_params(self):
        data_sparse = sparse.csr_matrix(self.data_item.values.astype(np.float))
        similarities = cosine_similarity(data_sparse.transpose())
        sim = pd.DataFrame(similarities, self.data_item.columns, self.data_item.columns)
        sim.columns = [int(i) for i in sim.columns]
        sim.index = [int(i) for i in sim.index]
        sim = pd.DataFrame(preprocessing.StandardScaler().fit(sim).transform(sim),
                           columns=sim.columns, index=sim.index)
        des = sim.describe().T
        return des['mean'].mean(), sim

    def get_user_projects(self, user_index, data_items):

        known_user_likes = data_items.loc[user_index]
        known_user_likes = known_user_likes[known_user_likes > 0].index.values
        return known_user_likes

    def get_recommendations_item_item(self, user_index, interactions ,test_items, k_similar):
        user_projects = self.sim_matrix[interactions]
        neighbourhood_size = 100
        data_neighbours = pd.DataFrame(0, user_projects.columns, range(1, neighbourhood_size + 1))
        for i in range(0, len(user_projects.columns)):
            data_neighbours.iloc[i, :neighbourhood_size] = user_projects.iloc[0:, i].sort_values(0, False)[
                                                           :neighbourhood_size].index
        # Construct the neighbourhood from the most similar items to the
        # ones our user has already liked.
        most_similar_to_likes = data_neighbours.loc[interactions]
        similar_list = most_similar_to_likes.values.tolist()
        similar_list = list(set([item for sublist in similar_list for item in sublist]))
        neighbourhood = self.sim_matrix[similar_list].loc[similar_list]

        user_vector = self.data_item.loc[user_index].loc[similar_list]
        score = neighbourhood.dot(user_vector).div(neighbourhood.sum(1))
        score = score.loc[np.intersect1d(score.index ,test_items)]

        recommended_projects_scores = score.astype(float).nlargest(k_similar)
        return recommended_projects_scores

    def get_recommendations_popularity(self, user_index, known_user_projects, k):
        projects_score = self.projects_popularity_scores.drop(known_user_projects)
        return list(projects_score.nlargest(k).index)

    def get_recommendations_content(known_user_projects, k_similar, user_test, sim, movies_df):
      user_projects = sim[known_user_projects]
      user_projects = pd.DataFrame(0, columns=user_projects.columns, index=user_projects.index)
      neighbourhood_size = movies_df.index.size
      data_neighbours = pd.DataFrame(0, user_projects.columns, range(1, neighbourhood_size + 1))
      for i in range(0, len(user_projects.columns)):
          data_neighbours.iloc[i, :neighbourhood_size] = user_projects.iloc[0:, i].sort_values(0, False)[:neighbourhood_size].index
      most_similar_to_likes = data_neighbours.loc[known_user_projects]
      similar_list = most_similar_to_likes.values.tolist()
      similar_list = list(set([item for sublist in similar_list for item in sublist]))
      data_items_col = user_test.values
      similar_list = [x for x in similar_list if x in data_items_col]

      data_matrix = sim.loc[known_user_projects]
      data_matrix = data_matrix[[p for p in similar_list]]
      # for recommendation:
      score = data_matrix.mean()
      score = score[score > 0]  # filter out score = 0
      relevant_projects_scores = score.nlargest(k_similar)
      return relevant_projects_scores

    def get_recommendations_dict(self, user_index, k):
        known_user_projects = self.get_user_projects(user_index, self.data_item)
        test = self.test_negative.iloc[user_index].to_list()
        # explained_content = self.get_recommendations_content(known_user_projects, k)
        explained_popularity = self.get_recommendations_popularity(user_index, known_user_projects, k)
        explained_item_item = self.get_recommendations_item_item(user_index, known_user_projects ,test, k)
        explained_user_user = self.get_recommendations_user_user(user_index, known_user_projects,test, k)
        return {'item_item': explained_item_item, 'user_user': explained_user_user, 'popularity': explained_popularity}

    def get_recommendations_dict_ordered(self, user_index, k):
        known_user_projects = self.get_user_projects(user_index, self.data_item)
        nais_predictions = self.nais_predictions.iloc[user_index]['predicted_list'][:k]
        if self.dataset=='ml-1m':
          explained_content = self.content_predictions.iloc[user_index]['predicted_list'][:k]
        else:
          explained_content =[]
        explained_popularity = self.get_recommendations_popularity(user_index, known_user_projects, k)
        explained_popularity = [str(i) for i in explained_popularity ]
        explained_user_user = self.user_user_predictions.iloc[user_index]['predicted_list'][:k]
        explained_item_item = self.item_item_predictions.iloc[user_index]['predicted_list'][:k]
        projects = {}
        for project in nais_predictions:
            """
            priority:
            - user_user, item_item, features.
            - if non - popularity.
            - if still non - general 
            """
            max_index = {'popularity': 100, 'user_user': 100, 'item_item': 100, 'content': 100, 'general': 100}
            if project in explained_user_user:
                max_index['user_user'] = explained_user_user.index(project)
            if project in explained_item_item:
                max_index['item_item'] = explained_item_item.index(project)
            if project in explained_content:
                max_index['content'] = explained_content.index(project)
            max_value = min(max_index.values())
            if max_value == 100:
                if project in explained_popularity:
                    max_index['popularity'] = explained_popularity.index(project)
                else:
                    max_index['general'] = 0  # smaller than -100
            max_value = min(max_index.values())
            max_key = [k for k in max_index if max_index[k] == max_value]

            projects[project] = max_key[0]
        # transform projects values into dictionary keys
        row = defaultdict(list)
        for key, value in projects.items():
            row[value].append(key)
        row['nais_predictions'] = nais_predictions
        row['known_user_projects'] = known_user_projects
        row['ranked_item'] = self.item_item_predictions.iloc[user_index]['ranked_item']
        return row

def create_predictions_and_choose(data_set):
    path = 'data'
    data = DataSet(path, data_set)
    test_negative = pd.read_csv(f'data/{data_set}.test.negative', delimiter='\t', header=None)
    test_negative[['user_id']] = test_negative.apply(lambda x: int(x[0].split(',')[0].replace('(', '')), 1)
    test_negative[[100]] = test_negative.apply(lambda x: int(x[0].split(',')[1].replace(')', '')), 1)
    test_negative = test_negative.drop(columns=[0, 'user_id'])
    R = RecommendationsAlgorithms(pd.DataFrame.sparse.from_spmatrix(data.trainMatrix), test_negative, dataset=data_set)
    pred_user_user = []
    pred_item_item = []
    k = 20
    for user in R.data_item.index:
        a = R.get_recommendations_dict(user, k)
        pred_item_item.append([user, test_negative.loc[user][100], a['item_item'].index.values, a['item_item'].values])
        pred_user_user.append([user, test_negative.loc[user][100], a['user_user'].index.values, a['user_user'].values])
    # outputting the tables to compare the models
    df_item_item = pd.DataFrame(pred_item_item, columns=['user', 'ranked_item', 'predicted_list', 'predicted_score'])
    df_user_user = pd.DataFrame(pred_user_user, columns=['user', 'ranked_item', 'predicted_list', 'predicted_score'])
    df_item_item.to_csv(f'predictions/{data_set}/item_item.csv', index=False)
    df_user_user.to_csv(f'predictions/{data_set}/user_user.csv', index=False)
    # choose the best algorithm to each Item
    R.read_predictions(data_set)
    for i in R.data_item.index:
        output = output.append(R.get_recommendations_dict_ordered(i, k), ignore_index=True)
    return output


if __name__ == "__main__":

    # ml Train all models and get predictions
    data_set = 'ml-1m'
    user_recommendations_per_alg = create_predictions_and_choose(data_set)
    user_recommendations_per_alg.to_csv(f'predictions/{data_set}/output.csv', index=False)