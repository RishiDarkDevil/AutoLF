# ---------------------------------------------- import necessary libraries

# To prevent warning errors in pandas DataFrames
import warnings
warnings.filterwarnings('ignore')

# general
import os
from functools import partial

# matrix manipulation
import numpy as np

# data handling
import pandas as pd

# webapp interface
import streamlit as st

# markmap visualization
from streamlit_markmap import markmap

# the model infos
from cluster_sentence.model_info import MODEL_PREPROCESSING, MODEL_BUILD_FUNCTIONS, MODEL_POSTPROCESSING

# embedding extractor
from cluster_sentence.sentence_embedding import EmbeddingExtractor

# embedding clusterer
from cluster_sentence.clustering import SentenceCluster

# clustering infos
from cluster_sentence.clustering_info import CLUSTERING_FUNCTIONS

# all the customizers
from cluster_sentence.customization_utils import (sentence_embedding_customizer, 
                                                  embedding_clustering_customizer, 
                                                  nested_dataframe_customizer)

# ---------------------------------------------- Page Layout

st.set_page_config(layout="wide")

# ---------------------------------------------- session state objects

# stores the label names
if 'label_names' not in st.session_state:
    st.session_state.label_names = list()

# stores the dataset
if 'data' not in st.session_state:
    st.session_state.data = list()

# stores the text column name
if 'text_column_name' not in st.session_state:
    st.session_state.text_column_name = list()

# stores the empty embedding extractor
if 'sent_emb_extractor' not in st.session_state:
    st.session_state.sent_emb_extractor = EmbeddingExtractor(None, None, None)

# stores the empty embedding extractor
if 'sent_emb_clusterer' not in st.session_state:
    st.session_state.sent_emb_clusterer = SentenceCluster(None)

# stores the clustering arguments
if 'clustering_args' not in st.session_state:
    st.session_state.clustering_args = {
        cluster_strat: {} for cluster_strat in CLUSTERING_FUNCTIONS.keys()
    }

# prepare data button state
if 'prep_data_button' not in st.session_state:
    st.session_state.prep_data_button = False

# cluster data button state
if 'clust_data_button' not in st.session_state:
    st.session_state.clust_data_button = False

# ---------------------------------------------- The Interface

# cola - set parameters and input data
# colb - labelling and rest of workflow
cola, colb = st.columns([0.8,1])

# ---------------------------------------------- Data Input and User Preferences

with cola:

    # title and intro to the wabapp
    st.title('AutoLF🏷: Interactive Labeller')
    st.write('⚡️ Create Labelled Datasets for your Machine Learning Model with Ease!')

    # upload the dataset
    up_data = st.file_uploader("👨🏻‍💻 Upload Dataset File", type={"csv", "txt"})
    if up_data is not None:

        # read csv dataset file
        st.session_state.data = pd.read_csv(up_data)
        
        # display dataset
        st.dataframe(st.session_state.data, use_container_width=True)

        # split into 2 columns
        col1, col2, col3, col4 = st.columns([1.2, 1, 0.5, 0.5])

        # select the column of dataset to label
        with col1:
            st.session_state.text_column_name = st.selectbox(
                '💬 Select Text Column of the Dataset',
                st.session_state.data.columns
                )
        
        # the labels to be used
        with col2:

            # enter label name
            label_name = st.text_input('🔖 Enter Label Name', value="", placeholder = 'Click Add to Update Labels')

        with col3:
            st.write('')
            st.write('')
            # add label button
            add_label = st.button('➕ Add', use_container_width=True)

        with col4:
            st.write('')
            st.write('')
            # reset label button
            reset_label = st.button('⎌ Reset', use_container_width=True)
            
        # add label
        if add_label:
            if label_name:
                st.session_state.label_names.append(label_name)
        
        # reset label list
        if reset_label:
            st.session_state.label_names.clear()

        col5, col6 = st.columns([2.2, 1])

        with col6:
            # display label names
            if len(st.session_state.label_names):
                st.dataframe(pd.DataFrame({'Label Names': st.session_state.label_names}), use_container_width=True)
            
            # start extracting the embeddings from the data for labelling ease
            st.button('🪄 Prepare Data', use_container_width=True, key='prep_data_button')
        
        with col5:

            # ---------------------------------------------- Customization to the Sentence Embedding Model

            # Contains all the customization option for the Sentence Transformer
            customize_sent_emb_modal = st.expander('⚙️ Sentence Embedding Customization')

            # add all the customization UI
            sentence_embedding_customizer(customize_sent_emb_modal)

            # ---------------------------------------------- Customization to the Sentence Embedding Clustering Model

            # Contains all the customization option for the Clustering Strategy
            customize_emb_clust_modal = st.expander('⚙️ Embedding Clustering Customization')

            # add all the customization UI
            embedding_clustering_customizer(customize_emb_clust_modal)

# ---------------------------------------------- Data Visualization

with colb:

    # ---------------------------------------------- Prepare Data
    if st.session_state.prep_data_button:
        
        # ---------------------------------------------- Embedding Extraction

        # Load the Sentence Embedding Model
        with st.spinner('Loading Sentence Embedding Model...'):

            # Load the embedding extractor
            st.session_state.sent_emb_extractor = EmbeddingExtractor(
                partial(
                    MODEL_BUILD_FUNCTIONS[st.session_state.sent_emb_model_name][st.session_state.sent_emb_backbone_name]['build_function'](
                        **MODEL_BUILD_FUNCTIONS[st.session_state.sent_emb_model_name][st.session_state.sent_emb_backbone_name]['args']
                    ).encode,
                    batch_size=st.session_state.sent_emb_batch_size,
                    convert_to_tensor=True
                ),
                MODEL_PREPROCESSING[st.session_state.sent_emb_model_name][st.session_state.sent_emb_backbone_name],
                MODEL_POSTPROCESSING[st.session_state.sent_emb_model_name][st.session_state.sent_emb_backbone_name]
            )
        
        with st.spinner('Extracting Sentence Embeddings...'):

            # Extract the Embeddings
            st.session_state.sent_emb_extractor.extract_embeddings(st.session_state.data[st.session_state.text_column_name].to_list())
        

# ---------------------------------------------- Embedding Clustering

# the cluster data button
if st.session_state.sent_emb_extractor.embeddings is not None:
    with col6:
        st.button('💡 Cluster Data', use_container_width=True, key='clust_data_button')

with colb:

    if st.session_state.clust_data_button:

        # clustering the embeddings
        with st.spinner('Clustering Sentences...'):
            st.session_state.sent_emb_clusterer = SentenceCluster(st.session_state.clust_strat_name)

            st.session_state.sent_emb_clusterer.cluster(
                st.session_state.sent_emb_extractor.embeddings / np.linalg.norm(st.session_state.sent_emb_extractor.embeddings, axis=1, keepdims=True), 
                st.session_state.clustering_args[st.session_state.clust_strat_name])
        
        ## create tree from clustered data --> For Markmap
        # with st.spinner('Creating Interactive Tree...'):
            # st.session_state.sent_emb_clusterer.convert_tree_to_markmap(st.session_state.data[st.session_state.text_column_name])

    ## markmap display --> For Markmap
    # markmap(st.session_state.sent_emb_clusterer.markmap_text,height=600)

        # create tree dataframe from clustered data
        with st.spinner('Creating Interactive Data Tree...'):
            st.session_state.sent_emb_clusterer.convert_tree_to_dataframe(st.session_state.data[st.session_state.text_column_name])

    # display tree dataframe
    tree_dataframe = nested_dataframe_customizer(st.session_state.sent_emb_clusterer.tree_dataframe)

with cola:
    st.write(tree_dataframe['selected_rows'])

    # import string

    # import numpy as np
    # import pandas as pd
    # import streamlit as st
    # from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    # # st.set_page_config(layout='wide')

    # def display_table(df: pd.DataFrame) -> AgGrid:
    #     # Configure AgGrid options
    #     gb = GridOptionsBuilder.from_dataframe(df)
    #     gb.configure_selection('multiple', use_checkbox=True) 
    #     return AgGrid(
    #         df,
    #         gridOptions=gb.build(),
    #         # this override the default VALUE_CHANGED
    #         update_mode=GridUpdateMode.MODEL_CHANGED
    #     )


    # # Define dummy data
    # rng = np.random.default_rng(2021)
    # N_SAMPLES = 100
    # N_FEATURES = 10
    # df = pd.DataFrame(rng.integers(0, N_SAMPLES, size=(
    #     N_SAMPLES, N_FEATURES)), columns=list(string.ascii_uppercase[:N_FEATURES]))

    # # Display data and selected rows
    # left, right = st.columns(2)
    # with left:
    #     st.info("Select rows to be deleted")
    #     response = display_table(df)
    # with right:
    #     st.warning("Rows selected for deletion")
    #     rows_to_delete = pd.DataFrame(response['selected_rows'])
    #     st.write(rows_to_delete)

    # # Delete rows on button press
    # if st.button("Delete rows") and not rows_to_delete.empty:
    #     # Lookup table is needed because AgGrid does not return rows indices
    #     lookup = df.merge(rows_to_delete, on=list(df.columns), how='left', indicator=True)
    #     _df = df.drop(lookup[lookup['_merge'] == 'both'].index)
    #     st.success('Rows deleted')
    #     st.write(_df)