# ---------------------------------------------- import necessary libraries

# matrix manipulation
import numpy as np

# Streamlit WebApp
import streamlit as st

# community clustering
from sentence_transformers.util import community_detection

# heirarchical clustering
from sklearn.cluster import AgglomerativeClustering

# agglomerative clustering helper functions
from .utils import create_linkage_matrix, create_tree_from_linkage

# ---------------------------------------------- Clustering Functions Argument and UI for user input

# the customization modal for the function args included in `CLUSTERING_FUNCTIONS`
# update the clustering args dictionary too with the user input values through the user input values
def add_cluster_args_and_add_customization(cluster_func_name, customize_modal, args_dict):
    '''
    Adds the arguments to the UI which are required to be input to the functions in `CLUSTERING_FUNCTIONS`
    :cluster_func_name: the name of the clustering function for which the `customize_modal` and `args_dict` will be updated
    :customize_modal: the expander modal to which the argument entry UI should be added
    :args_dict: the argument dictionary which should store all the argument value input
    '''

    # ---------------------------------------------- Agglomerative Clustering
    if cluster_func_name == 'Agglomerative Clustering':

        # number of clusters
        args_dict['n_clusters'] = None

        # metric
        args_dict['metric'] = customize_modal.selectbox(
            'Metric',
            ['cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
            index=0,
            help='The metric to be used in Agglomerative Clustering'
        )

        # linkage
        args_dict['linkage'] = customize_modal.selectbox(
            'Linkage',
            ['average', 'ward', 'complete', 'single'],
            index=0,
            help='The Linkage to be used in Agglomerative Clustering'
        )

        # distance threshold
        args_dict['distance_threshold'] = customize_modal.slider(
            'Distance Threshold',
            0.0, 1.0, 0.5, 0.01, '%f',
            help='The Distance Threshold to be used in Agglomerative Clustering'
        )
    
    # ---------------------------------------------- Community Clustering
    if cluster_func_name == 'Community Clustering':

        # minimum community size
        args_dict['min_community_size'] = customize_modal.number_input(
            'Minimum Community Size', 
            1, None, 1, 1, '%d',  
            help='Minimum Size of the Cluster to be kept'
        )

        # merge threshold
        args_dict['threshold'] = customize_modal.slider(
            'Threshold',
            0.0, 1.0, 0.6, 0.01, '%f',
            help='The similarity threshold above which will present in same cluster '
        )

def agl_cluster(embeddings, **kwargs):
    '''
    :embeddings: the embedding matrix of shape: number of sentences x embedding dimension
    '''
    # initialize Agglomerative Clustering with given arguments
    clustering_model = AgglomerativeClustering(**kwargs)
    # fit the model
    clustering_model.fit(embeddings)

    return clustering_model

# ---------------------------------------------- Clustering Functions

# the clustering functions supported
CLUSTERING_FUNCTIONS = {
    'Community Clustering': community_detection,
    'Agglomerative Clustering': agl_cluster
}

# ---------------------------------------------- Cluster Function's Output Post-Processing

# create tree structure from the cluster functions output

# ---------------------------------------------- Community Clustering

# community clustering output post process
def community_clustering_create_tree(clusters):
    '''
    converts the clusters to tree for Community Clustering
    :clusters: list of list of sentence ids where each inner list is a cluster
    '''
    # the total number of sentences clustered
    num_sents = sum([len(sent_ids) for sent_ids in clusters])

    # convert the clusters into a D3.js type tree
    tree = {
        'name': num_sents + len(clusters), 
        'parent': None,
        'children': list()
        }
    for idx, sent_ids in dict(list(enumerate(clusters))).items():
        node = {
            'name': num_sents + idx,
            'parent': len(clusters),
            'children': [
                {
                    'name': sent_id,
                    'parent': idx,
                    'children': list()
                } for sent_id in sent_ids
            ]
        }
        tree['children'].append(node)

    return tree

# ---------------------------------------------- Agglomerative Clustering

# community clustering output post process
def agglomerative_clustering_create_tree(model):
    '''
    converts the clusters to tree for Agglomerative Clustering
    :model: the fitted agglomerative clustering model
    '''
    # get the linkage matrix of the trained Agglomerative Clustering Model
    linkage_matrix = create_linkage_matrix(model)

    # Convert `linkage_matrix` to tree
    tree = create_tree_from_linkage(linkage_matrix)

    return tree

# ---------------------------------------------- Clustering Post processing Functions

# the output processing for the clustering functions
CLUSTERING_POST_PROCESSING = {
    'Community Clustering': community_clustering_create_tree,
    'Agglomerative Clustering': agglomerative_clustering_create_tree
}
