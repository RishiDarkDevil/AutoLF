# ---------------------------------------------- import necessary libraries

# Streamlit WebApp
import streamlit as st

# the model details
from .model_info import MODEL_PREPROCESSING, MODEL_BUILD_FUNCTIONS, MODEL_POSTPROCESSING

# the clustering details
from .clustering_info import CLUSTERING_FUNCTIONS

# ---------------------------------------------- Add Customizer Expanders

def sentence_embedding_customizer(customize_modal):
    '''
    Adds the UI for the sentence embedding customizer to the expander `modal`
    :customize_modal: the expander modal to which the other customization options needs to be added
    '''
    # Select Model for Sentence Embedding
    customize_modal.selectbox(
        'Model',
        MODEL_PREPROCESSING.keys(),
        index=0, 
        key='sent_emb_model_name',
        help='The Sentence Embedding Model to be used for Clustering'
        )
    
    # Select Backbone for the Sentence Embedding Model
    customize_modal.selectbox(
        'Backbone',
        MODEL_PREPROCESSING[st.session_state.sent_emb_model_name].keys(),
        index=0,
        key='sent_emb_backbone_name',
        help='The Backbone to use with the selected Sentence Embedding Model'
        )

    # Batch Size for the Sentence Embedding Model
    customize_modal.number_input(
        'Batch Size', 
        1, None, 2048, 1, '%d', 
        key='sent_emb_batch_size', 
        help='Batch Size to be used by the Sentence Transformer Model') # , disabled=st.session_state.rand
    
def embedding_clustering_customizer(customize_modal):
    '''
    Adds the UI for the embedding clustering customizer to the expander `modal`
    :customize_modal: the expander modal to which the other customization options needs to be added
    '''
    # Select Clustering Strategy for Sentence Embedding
    customize_modal.selectbox(
        'Clustering Strategy',
        CLUSTERING_FUNCTIONS.keys(),
        index=1, 
        key='clust_strat_name',
        help='The Clustering Strategy for the Sentence Embeddings'
        )
    
    # Agglomerative Clustering Customization
    if st.session_state.clust_strat_name == 'Agglomerative Clustering':

        # number of clusters
        st.session_state.clustering_args[st.session_state.clust_strat_name]['n_clusters'] = None

        # metric
        st.session_state.clustering_args[st.session_state.clust_strat_name]['metric'] = customize_modal.selectbox(
            'Metric',
            ['cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
            index=0,
            help='The metric to be used in Agglomerative Clustering'
        )

        # linkage
        st.session_state.clustering_args[st.session_state.clust_strat_name]['linkage'] = customize_modal.selectbox(
            'Linkage',
            ['average', 'ward', 'complete', 'single'],
            index=0,
            help='The Linkage to be used in Agglomerative Clustering'
        )

        # distance threshold
        st.session_state.clustering_args[st.session_state.clust_strat_name]['distance_threshold'] = customize_modal.slider(
            'Distance Threshold',
            0.0, 1.0, 0.5, 0.01, '%f',
            help='The Distance Threshold to be used in Agglomerative Clustering'
        )
    
    # Community Clustering Customization
    if st.session_state.clust_strat_name == 'Community Clustering':

        # minimum community size
        st.session_state.clustering_args[st.session_state.clust_strat_name]['min_community_size'] = customize_modal.number_input(
            'Minimum Community Size', 
            1, None, 1, 1, '%d',  
            help='Minimum Size of the Cluster to be kept'
        )

        # linkage
        st.session_state.clustering_args[st.session_state.clust_strat_name]['threshold'] = customize_modal.slider(
            'Threshold',
            0.0, 1.0, 0.6, 0.01, '%f',
            help='The similarity threshold above which will present in same cluster '
        )
    
