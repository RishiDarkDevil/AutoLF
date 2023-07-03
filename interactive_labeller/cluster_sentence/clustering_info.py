# ---------------------------------------------- import necessary libraries

# Streamlit WebApp
import streamlit as st

# community clustering
from sentence_transformers.util import community_detection

# heirarchical clustering
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------- Clustering Functions

CLUSTERING_FUNCTIONS = {
    'Community Clustering': community_detection,
    'Agglomerative Clustering': AgglomerativeClustering
}

