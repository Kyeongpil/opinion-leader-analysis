# coding: utf-8
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from math import log10
from random import shuffle
import numpy as np
import networkx as nx
import pickle


with open("./preprocessed_bitcoin.pkl", 'rb') as f:
    data = pickle.load(f)

voca2idx = {w: i for i, w in enumerate(data['voca'])}
voca = data['voca']


# HITS algorithm
hubs, user_score = nx.hits(data['user_network'], max_iter=500)
total_user_num = len(data['user_network'].nodes())
print(total_user_num, len(data['posts']))
users = sorted(user_score, key=user_score.get, reverse=True)
score_sum = sum(user_score.values())

# 전체 유저 분석!
tdm = dok_matrix((len(data['posts']), len(voca)), dtype=np.float32)
for i, post in enumerate(data['posts']):
    for word in post:
        tdm[i, voca2idx[word]] += 1
        
tdm = normalize(tdm)
tdm = tdm.tocsr()

K = 10
nmf = NMF(n_components=K, alpha=0.1, max_iter=500)
nmf.fit(tdm)
H_total = nmf.components_


proportions = [0.0001, 0.0005, 0.001]
proportions.extend([0.005 * i for i in range(1, 24)])
fwrite = open("./proportion_random_result.csv", "w")
fwrite.write("proportion, trial, user_num, user_posts_num, similarity_avg\n")

trials = 10
for proportion in proportions:
    mean_random_num = 0
    mean_user_posts_num = 0
    mean_similarity_average = 0
    for trial in range(trials):
        while True:
            rand_index = [i for i in range(len(users))]
            shuffle(rand_index)

            random_num = round(len(users)*proportion)
            random_users = [users[i] for i in rand_index[:random_num]]

            user_posts = []
            for user in random_users:
                for post in data['user_posts'][user]:
                    user_posts.append(post)

            user_posts_num = len(user_posts)

            tdm = dok_matrix((len(user_posts), len(voca)), dtype=np.float32)
            for i, post in enumerate(user_posts):
                for word in post:
                    tdm[i, voca2idx[word]] += 1

            tdm = normalize(tdm)
            tdm = tdm.tocsr()

            try:
                nmf = NMF(n_components=K, alpha=0.1, max_iter=500)
                nmf.fit(tdm)
            except:
                continue
            H_random = nmf.components_


            # hungarian algorithm
            distances = pairwise_distances(H_total, H_random, metric='cosine')
            _, top_indices = linear_sum_assignment(distances)

            similarity_average = 0
            for k in range(K):
                similarity = cosine_similarity(H_random[top_indices[k]].reshape(1, -1), H_total[k].reshape(1,-1))[0, 0] 
                similarity_average += similarity

            similarity_average /= K
 
            fwrite.write("{}, {}, {}, {}, {}\n".format(proportion, trial, random_num, user_posts_num, similarity_average))

            break
        
        mean_random_num += random_num
        mean_user_posts_num += user_posts_num
        mean_similarity_average += similarity_average
    
    mean_random_num /= trials
    mean_user_posts_num /= trials
    mean_similarity_average /= trials
    print("{}%, mean_random_users: {:.2f}, mean_random_user_posts: {:.2f}, mean_random_similarity: {:.6f}".format(proportion*100,  mean_random_num, mean_user_posts_num, mean_similarity_average))

fwrite.close()
