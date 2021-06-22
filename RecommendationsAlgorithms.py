from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from itertools import chain
from BPR import BPR
from hybrid import Hybrid
from ContentBased import ContentBased
from collections import defaultdict
from Location_based_features import valid_recommendations, explainable_by_location, get_project_location_name
from sklearn import preprocessing
from num2words import num2words
from helper import shuffle_dict
from files import *
from explain import get_display_order_list
import pandas as pd
import json
import datetime
import operator

class RecommendationsAlgorithms:
    def __init__(self, data_item, projects_data, topics_associations, non_active_projects):
        self.data_item = data_item

        self.item_mean, self.sim_matrix = self.get_item_item_params()
        self.projects_popularity_scores = data_item.astype(bool).sum(axis=0)
        self.non_active_projects = non_active_projects
        self.projects_content = projects_data[
            ['name', 'time_per_participation', 'location_type', 'project_topic_1', 'project_topic_2']]
        self.projects_content = self.projects_content[self.projects_content.index.isin(data_item.columns)]
        self.content_similarity = self.calculate_content_similarity()  # project * project
        self.associations = topics_associations

    def train(self,epochs=235,learning_rate=0.010418970926384429, no_components=92, item_alpha=0.00043201423592654906):
        self.rec_algorithm = Hybrid(self.data_item, alpha=0.2,
                                    BPR=BPR(self.data_item, epochs=epochs, learning_rate=learning_rate,
                                            no_components=no_components, item_alpha=item_alpha),
                                    CB_HOT=ContentBased(self.data_item, None))
        self.model_knn = NearestNeighbors(100, 1.0, 'brute', 30, 'cosine')
        self.model_knn.fit(self.data_item)


    def calculate_content_similarity(self):
        content_similarity = self.projects_content
        hot = pd.get_dummies(content_similarity)
        similarities = cosine_similarity(hot)
        sim = pd.DataFrame(similarities, content_similarity.index, content_similarity.index)
        sim.columns = [int(i) for i in sim.columns]
        sim.index = [int(i) for i in sim.index]
        sim = pd.DataFrame(preprocessing.StandardScaler().fit(sim).transform(sim),
                           columns=sim.columns, index=sim.index)
        return sim

    def get_feature_similar(self, project, known_user_projects):
        feature_map = {'time_per_participation': 'time', 'location_type': 'location', 'project_topic_1': 'topic', 'project_topic_2': 'topic'}
        for p in known_user_projects:
            d = dict(self.projects_content.loc[p] == self.projects_content.loc[project])
            d = shuffle_dict(d)
            for f in d:
                if d[f]:
                    if f == 'name' or f == 'location_type':
                        pass
                    return feature_map[f]

    def get_recommendations_content(self, known_user_projects, k_similar, ip_address):
        relevant_known_projects = list(filter(lambda p: p in self.projects_content.index,
                                              known_user_projects))  # if the user projects history have content
        not_all_content = len(relevant_known_projects) != len(known_user_projects)
        if not_all_content:
            return []
        else:
            user_projects = self.content_similarity[known_user_projects]
            user_projects = pd.DataFrame(0, columns=user_projects.columns, index=user_projects.index)
            neighbourhood_size = self.projects_content.index.size
            data_neighbours = pd.DataFrame(0, user_projects.columns, range(1, neighbourhood_size + 1))
            for i in range(0, len(user_projects.columns)):
                data_neighbours.iloc[i, :neighbourhood_size] = user_projects.iloc[0:, i].sort_values(0, False)[
                                                               :neighbourhood_size].index
            most_similar_to_likes = data_neighbours.loc[known_user_projects]
            similar_list = most_similar_to_likes.values.tolist()
            similar_list = list(set([item for sublist in similar_list for item in sublist]))

            data_items_col = self.content_similarity.columns.to_list()
            similar_list = [x for x in similar_list if x in data_items_col]

            data_matrix = self.content_similarity.loc[known_user_projects]
            data_matrix = data_matrix[[p for p in similar_list]]
            # for recommendation:
            score = data_matrix.mean()
            score = score[score > 0]  # filter out score = 0
            score = valid_recommendations(score, ip_address)
            relevant_projects_scores = score.nlargest(k_similar)

            return relevant_projects_scores

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

    def find_k_similar_users(self, user_index, k=250):
        distances, indices = self.model_knn.kneighbors(
            self.data_item.iloc[user_index, :].values.reshape(1, -1), n_neighbors=k + 1)
        similarities = 1 - distances.flatten()
        return pd.Series(similarities, indices[0])

    def get_most_similar_to_recommendation(self, recommended_project, interactions):
        return self.sim_matrix[recommended_project].loc[interactions].nlargest(1).index.values[0]

    def get_recommendations_item_item(self, user_index, interactions, k_similar, ip_address):
        user_projects = self.sim_matrix[interactions]
        neighbourhood_size = k_similar
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
        for project in interactions:
            if project in score.index:
                score = score.drop(project)
        score = valid_recommendations(score, ip_address)

        recommended_projects_scores = score.astype(float).nlargest(k_similar)
        return recommended_projects_scores

    def get_recommendations_user_user(self, user_index, known_user_projects, k, ip_address):
        similar_users = self.find_k_similar_users(user_index)
        if user_index in similar_users.index:
            similar_users = similar_users.drop(user_index, 0)
        similar_projects = [self.get_user_projects(user, self.data_item) for user in similar_users.index]
        similar_projects = list(set(chain(*similar_projects)))  # get all the projects from the dataframe
        projects_scores = dict.fromkeys(similar_projects, 0)

        for s_project in similar_projects:
            for user in similar_users.index:
                projects_scores[s_project] += similar_users.loc[user] * self.data_item.loc[user][s_project]
        projects_scores_to_fit = np.array(list(projects_scores.values())).reshape(-1, 1)
        scaled_scores = preprocessing.StandardScaler().fit(projects_scores_to_fit).transform(projects_scores_to_fit)
        projects_scores = pd.DataFrame(scaled_scores, index=pd.Series(projects_scores).index)[0]
        for project in known_user_projects:
            if project in projects_scores.index:
                projects_scores = projects_scores.drop(project)
        score = valid_recommendations(projects_scores, ip_address)

        return score.astype(float).nlargest(k)

    def get_project_popularity_ordinal_number(self, project):
        n = list(self.projects_popularity_scores.sort_values(ascending=False).axes[0]).index(project) + 1
        return num2words(n, lang="en", to="ordinal_num")

    def get_recommendations_popularity(self, user_index, known_user_projects, k, ip_address):
        projects_score = self.projects_popularity_scores.drop(known_user_projects)
        score = valid_recommendations(projects_score, ip_address)
        return list(score.nlargest(k).index)

    def get_project_topic(self, project):
        user_topics = []
        user_topics.append(self.projects_content[self.projects_content.index == project].project_topic_1.to_list())
        user_topics.append(self.projects_content[self.projects_content.index == project].project_topic_2.to_list())
        user_topics = list(set([val for sublist in user_topics for val in sublist if str(val) != 'nan']))
        return user_topics

    def is_explained_by_associations(self, recommended_project, known_user_projects, get_user_association_topic=False):
        # get User's topics
        user_topics = []
        for project in known_user_projects:
            user_topics.append(self.get_project_topic(project))  # get all topics that the user liked
        user_topics = list(set([val for sublist in user_topics for val in sublist if str(val) != 'nan']))  # expend to one list
        recommended_topics = self.get_project_topic(recommended_project)  # getting the recommended topics
        associations_user = self.associations[self.associations['antecedents'].isin(user_topics)]  # getting consequent topics from pre trained associations rules
        associations_target = [tuple(x) for x in associations_user.to_numpy()]
        associations_target = [x for x in associations_target if x[1] not in user_topics]  # drop user's topics from different projects
        if len(np.intersect1d([x[1] for x in associations_target], recommended_topics)) > 0:
            associations_target = dict(associations_target)
            if get_user_association_topic:
                for topic in user_topics:
                    if associations_target.get(topic):
                        return topic, associations_target.get(topic)  # only appears once
            return 100
        return -100

    def is_explained_by_location(self, project, location_projects, ip_address):
        if project in location_projects and ip_address is not np.nan:
            return 99
        return -100

    def get_homepage_recommendations(self, algorithms, headers, recommendations):
        homepage_projects = []
        i = 0
        tooltips = []
        header = headers['general']
        exp_type = 'general'
        while len(homepage_projects) < 3:
            alg_dict = recommendations.get(algorithms[i])
            if alg_dict:
                if len(homepage_projects) == 0 and len(alg_dict['projects']) >= 3:  # all projects from the same header.
                    header = alg_dict['explanation_header']
                    exp_type = algorithms[i]
                homepage_projects.extend(alg_dict['projects'])
                tooltips.extend(alg_dict['tooltip'])
            i += 1
        homepage = {exp_type: {'explanation_header': header, 'projects': homepage_projects[:3], 'tooltip': tooltips[:3]}}
        return homepage

    def get_json_data(self, row, user, ip_address,for_evaluation=False):
        recommendations = {}
        row = pd.Series(row)
        list_ordered = get_display_order_list(row) # organize the projects groups and project considering the projects score and the explanation priority.
        headers = {'associations': 'Try projects with new topics', 'content': 'Projects with features that fit you', 'user_user': 'Others like you liked these projects',
                   'item_item': 'Similar to projects that you like', 'popularity': 'Hot on SciStarter', 'location': 'Projects near you.','general': 'Try something new'}
        homepage = ''
        if for_evaluation:
            experiment_group = 1
        else:
            experiment_group = get_user_group(user)
        if experiment_group:
            for alg in list_ordered:
                if alg in headers.keys():  # to ignore the user id and known_user_likes
                    if row[alg] and row[alg] == row[alg]:
                        # if len(homepage) == 0 and len(row[alg]) > 2:
                        #     homepage = alg
                        if alg == 'item_item':
                            tooltip = []
                            for project in row[alg]:
                                user_project_id = self.get_most_similar_to_recommendation(project,
                                                                                          row['known_user_projects'])
                                try:
                                    user_project_name = projects_id_name_publish.loc[user_project_id]['name']
                                    recommended_name = projects_id_name_publish.loc[project]['name']
                                    tooltip.append(
                                        {project: f'People who liked {user_project_name} also liked {recommended_name}.'})
                                except:
                                    # Todo: log: Missing name for user_project_id or project
                                    tooltip.append({project: f''})
                        elif alg == 'associations':
                            tooltip = []
                            for project in row[alg]:
                                user_liked_topic, recommended_topic = self.is_explained_by_associations(project, row['known_user_projects'], True)
                                tooltip.append({
                                    project: f'People who did {user_liked_topic} projects liked {recommended_topic}.'})
                        elif alg == 'popularity':
                            tooltip = []
                            for project in row[alg]:
                                project_order = self.get_project_popularity_ordinal_number(project)
                                tooltip.append({
                                    project: f'This project is the {project_order} most popular project in SciStarter!'})
                        elif alg == 'content':
                            tooltip = []
                            for project in row[alg]:
                                try:
                                    feature = self.get_feature_similar(project,row['known_user_projects'])
                                    if feature == 'time':
                                        recommended_name = projects_id_name_publish.loc[project]['name']
                                        tooltip_content = f'{recommended_name} may fit your schedule.'
                                    elif feature == 'topic':
                                        topic = self.get_project_topic(project)[0]
                                        tooltip_content = f'You interested in {topic} in the past.'
                                    else:
                                        tooltip_content = ''
                                    tooltip.append({
                                        project: tooltip_content})

                                except:
                                    # Todo: log: Missing name for user_project_id or project
                                    tooltip.append({project: f''})

                        elif alg == 'location':
                            tooltip = []
                            for project in row[alg]:

                                try:
                                    # project_location_name = get_project_location_name(int(project))
                                    # recommended_name = projects_id_name_publish.loc[project]['name']
                                    tooltip_content = ''
                                    # if project_location_name == project_location_name:
                                    #     if not project_location_name.startswith('Unknown'):
                                    #         tooltip_content = f'{recommended_name} is done in {project_location_name}.'

                                    tooltip.append({
                                        project: tooltip_content})
                                except:
                                    # Todo: log: Missing name for user_project_id or project
                                    tooltip.append({project: f''})
                        else:
                            tooltip = []
                            for project in row[alg]:
                                tooltip_content = ''
                                tooltip.append({
                                    project: tooltip_content})
                        rec = {'explanation_header': headers[alg], 'projects': row[alg], 'tooltip': tooltip}
                        recommendations[alg] = rec

            homepage_dict = self.get_homepage_recommendations(list_ordered, headers, recommendations)
        else:
            recommendations = []
            homepage_dict = ''
            for alg in list_ordered:
                if row[alg] and row[alg] == row[alg]:
                    recommendations.extend(row[alg])

        user_dict = {
            'user': user,
            'group': get_user_group(user),
            'timestamp': str(datetime.datetime.now()),
            'homepage': homepage_dict,
            'recommendations': recommendations

        }
        return json.dumps(user_dict)
    def make_exp_type_three_projects(self,row,exp_type):

        '''
        need to remove the most popular project from exp_type that have more then three projects and have the highest number of projects.
        '''

        pop_list_sort = list(self.projects_popularity_scores.sort_values(ascending=False).axes[0])
        popularity_projects = row.get(exp_type)
        if popularity_projects is None:  # no need to take action
            return row
        keys_to_ignore = [exp_type]
        while len(row[exp_type]) < 3 and max(len(v) for k,v in row.items()) > 3 and len(keys_to_ignore) < len(row.keys()):
            s_row = pd.Series(row).drop(keys_to_ignore, errors='ignore')
            sorted_keys = sorted(dict(s_row), key=lambda k: len(dict(s_row)[k]), reverse=True)
            key = sorted_keys[0]
            if len(row[key]) >3:
                projects_rank_popularity = {}
                for project in row[key]:
                    n = pop_list_sort.index(project) + 1
                    projects_rank_popularity[project] = n
                tuple_to_transfer = min((v,k) for k,v in projects_rank_popularity.items())
                n = tuple_to_transfer[0]
                project_to_transfer = tuple_to_transfer[1]
                if n < 20:
                    row[key].remove(project_to_transfer)
                    row[exp_type].append(project_to_transfer)
                else:
                    keys_to_ignore.append(key)
            else:
                keys_to_ignore.append(key)
        return row



    def get_recommendations_dict(self, user, user_index, k, ip_address=None):
        known_user_projects = self.get_user_projects(user_index, self.data_item)
        recommended = self.rec_algorithm.get_recommendations(user_index, known_user_projects, k, ip_address)
        explained_location = explainable_by_location(recommended)
        explained_content = self.get_recommendations_content(known_user_projects, 20, ip_address)
        explained_popularity = self.get_recommendations_popularity(user_index, known_user_projects, 20, ip_address)
        # explained_user_user = self.get_recommendations_user_user(user_index, known_user_projects, 20, ip_address)
        explained_item_item = self.get_recommendations_item_item(user_index, known_user_projects, 20, ip_address)
        projects = {}
        for project in recommended:
            """
            priority:
            - explained_by_associations- if the project exists there he gets 100 score 
            - user_user, item_item, features.
            - if non - priority.
            - if still non - general 
            """
            associations_value = self.is_explained_by_associations(project, known_user_projects)
            location_value = self.is_explained_by_location(project, explained_location, ip_address)
            max_score = {'popularity': -100, 'user_user': -100, 'item_item': -100, 'content': -100, 'general': -100,
                         'associations': associations_value, 'location': location_value}
            # if project in explained_user_user:
            #     max_score['user_user'] = explained_user_user[project]
            if project in explained_item_item:
                max_score['item_item'] = explained_item_item[project]
            if project in explained_content:
                max_score['content'] = explained_content[project]
            max_value = max(max_score.values())
            if max_value == - 100:
                if project in explained_popularity:
                    max_score['popularity'] = explained_popularity.index(project)
                else:
                    max_score['general'] = -1  # bigger than -100
            max_value = max(max_score.values())
            max_key = [k for k in max_score if max_score[k] == max_value]

            projects[project] = max_key[0]
        # transform projects values into dictionary keys
        row = defaultdict(list)
        for key, value in projects.items():
            row[value].append(key)
        self.make_exp_type_three_projects(row,'popularity')
        row['recommended'] = recommended
        row['user'] = user
        row['known_user_projects'] = list(known_user_projects)
        if ip_address == ip_address:
            print()
        return row

# data = pd.read_csv('data/27.12.20/user_projects_ip.csv')
# data_items = data.drop(['user', 'ip'], 1)
# data_items.columns = data_items.columns.astype(int)
# data_items.reset_index(inplace=True, drop=True)
# recs_algs = RecommendationsAlgorithms(data_items)
# user ='da164c8d-5d09-5fe9-aa78-4d2ef06f6d99'
# # user_index = data[data.user ==user].index[0]
# #
# # recs_algs.get_recommendations_dict(user, user_index, 10)
#
