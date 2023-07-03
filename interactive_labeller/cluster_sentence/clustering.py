# ---------------------------------------------- import necessary libraries

from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------- Clustering

class SentenceCluster:

    def __init__(self, embeddings, cluster_function):
        '''
        The Sentence Embedding Clustering Model
        :embeddings: the embedding matrix of shape: number of sentences x embedding dimension
        :cluster_function: should take in the embedding and output clustered data 
                        (currently includes `Community Clustering` and `Agglomerative Clustering`)
        '''
        self.embeddings = embeddings
        self.cluster_strategy = cluster_function
        
