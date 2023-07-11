# ---------------------------------------------- import necessary libraries

# data handling
import pandas as pd

# get the clustering functions and corresponding post processing available
from .clustering_info import CLUSTERING_FUNCTIONS, CLUSTERING_POST_PROCESSING

# utils
from .utils import tree_to_markmap, tree_to_dataframe

# ---------------------------------------------- Clustering

class SentenceCluster:

    def __init__(self, cluster_function_name):
        '''
        The Sentence Embedding Clustering Model
        :cluster_function_name: should take in the embedding and output clustered data 
                        (currently includes `Community Clustering` and `Agglomerative Clustering`)
        '''
        self.cluster_func_name = cluster_function_name
        self.cluster_tree = None
        # For Markmap
        self.markmap_text = 'AutoLF - By RishiDarkDevil'
        # For Tree DataFrame
        self.tree_dataframe = pd.DataFrame({
            'Hierarchy': ['AutoLF', 'AutoLF/Creator'], 'Sentence': ['Data Programming', 'RishiDarkDevil']
        })

    def cluster(self, embeddings, args):
        '''
        clusters the `embeddings` with the `self.cluster_function`
        :embeddings: the embedding matrix of shape: number of sentences x embedding dimension
        :args: arguments to be used for clustering apart from embeddings
        '''
        clusters = CLUSTERING_FUNCTIONS[self.cluster_func_name](embeddings, **args)
        self.cluster_tree = CLUSTERING_POST_PROCESSING[self.cluster_func_name](clusters)

    def convert_tree_to_markmap(self, sentences, num_levels = 200):
        '''
        dictionary to markmap text. Can be used if markmap is intended to be used
        :sentences: list of sentences which are to be clustered
        :num_levels: number of levels of the cluster tree to print
        '''
        if self.cluster_tree is not None:
            # stores the markmap text created from the tree
            self.markmap_text = tree_to_markmap(self.cluster_tree, num_levels, sentences)
    
    def convert_tree_to_dataframe(self, sentences, num_levels = 200):
        '''
        convert the hierarchical data into a tree dataframe for easier visualization
        :sentences: list of sentences which are to be clustered
        :num_levels: number of levels of the cluster tree to print
        '''
        if self.cluster_tree is not None:
            # stores the markmap text created from the tree
            self.tree_dataframe = tree_to_dataframe(self.cluster_tree, num_levels, sentences)

    

        
