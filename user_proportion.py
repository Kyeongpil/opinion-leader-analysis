# coding: utf-8
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from math import log10
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
proportions.extend([i * 0.005 for i in range(1, 24)])
fwrite = open("./proportion_result.csv", "w")
fwrite.write("proportion, user_num, user_posts_num, similarity_avg\n")
for proportion in proportions:
    # 상위 유저 분석!
    top_num = round(len(users)*proportion)
    top_users = users[:top_num]

    user_posts = []
    for user in top_users:
        for post in data['user_posts'][user]:
            user_posts.append(post)
            
    top_user_posts_num = len(user_posts)

    tdm = dok_matrix((len(user_posts), len(voca)), dtype=np.float32)
    for i, post in enumerate(user_posts):
        for word in post:
            tdm[i, voca2idx[word]] += 1

    tdm = normalize(tdm)
    tdm = tdm.tocsr()

    nmf = NMF(n_components=K, alpha=0.1, max_iter=500)
    nmf.fit(tdm)
    H_top = nmf.components_


    # hungarian algorithm
    top_distances = pairwise_distances(H_total, H_top, metric='cosine')
    _, top_indices = linear_sum_assignment(top_distances)

    similarity_average = 0
    for k in range(K):
        similarity = cosine_similarity(H_top[top_indices[k]].reshape(1, -1), H_total[k].reshape(1,-1))[0, 0] 
        similarity_average += similarity
        
    similarity_average /= K
    
    print("{}%, top_users: {}, top_user_posts: {}, top_similarity: {}".format(proportion*100, top_num, top_user_posts_num, similarity_average))    
    fwrite.write("{}, {}, {}, {}\n".format(proportion, top_num, top_user_posts_num, similarity_average))

fwrite.close()
