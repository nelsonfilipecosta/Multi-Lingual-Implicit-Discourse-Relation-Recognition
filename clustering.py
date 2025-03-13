import ast
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment

path = 'Data/PDTB-3.0/pdtb_3_embeddings_all-MiniLM-L6-v2.csv'
columns = ['relation', 'sense1', 'multi_sense1',  'embeddings']

num_clusters = 4

RELATION_TYPE = sys.argv[1]
if RELATION_TYPE not in ['explicit', 'implicit']:
    print('Type a valid relation type: explicit or implicit.')
    exit()

ALGORITHM = sys.argv[2]
if ALGORITHM not in ['kmeans', 'gmm', 'hdbscan', 'agglomerative', 'spectral']:
    print('Type a valid relation type: kmeans, gmm, hdbscan, agglomerative or spectral.')
    exit()

def cluster_mapping_dictionary(clusters, senses):
    # compute frequency table
    contingency_table = pd.crosstab(clusters, senses)
    # solve linear sum assignment problem
    row_ind, col_ind = linear_sum_assignment(-contingency_table.values)
    # create mapping dictionary
    cluster_map = {cluster: category for cluster, category in zip(row_ind, contingency_table.columns[col_ind])}
    return cluster_map

start_time = time.time()
print(f'Clustering with {ALGORITHM}...')

complete_df = pd.read_csv(path, usecols=columns)

if RELATION_TYPE == 'explicit':
    df = complete_df[complete_df['relation'] == 'Explicit'].copy()
else:
    df = complete_df[complete_df['relation'] == 'Implicit'].copy()

if ALGORITHM == 'kmeans':
    model = KMeans(n_clusters=num_clusters, algorithm='lloyd', init='k-means++')
elif ALGORITHM == 'gmm':
    model = GaussianMixture(n_components=num_clusters)
elif ALGORITHM == 'hdbscan':
    model = HDBSCAN(min_cluster_size=5)
elif ALGORITHM == 'agglomerative':
    model = AgglomerativeClustering(n_clusters=num_clusters)
elif ALGORITHM == 'spectral':
    model = SpectralClustering(n_clusters=num_clusters)

df['embeddings'] = df['embeddings'].apply(lambda x: ast.literal_eval(x))

embeddings = np.vstack(df['embeddings'].values)

print(embeddings.shape)

# pca = PCA(n_components=10)
# embeddings = pca.fit_transform(embeddings)
# print(embeddings.shape)

df['cluster'] = model.fit_predict(embeddings)

cluster_map = cluster_mapping_dictionary(df['cluster'], df['sense1'])

df['predicted_category'] = df['cluster'].map(cluster_map)

accuracy = (df['predicted_category'] == df['sense1']).mean()*100

print(f'Completed in {(time.time()-start_time)/60:.2f} minutes with an accuracy of {accuracy:.2f}%.')