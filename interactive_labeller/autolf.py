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

# label tags
from streamlit_tags import st_tags

# the model infos
from cluster_sentence.model_info import MODEL_PREPROCESSING, MODEL_BUILD_FUNCTIONS, MODEL_POSTPROCESSING

# Aggrid Table update mode
from st_aggrid import GridUpdateMode

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

# to store the labelled tree
if 'labelled_tree_data' not in st.session_state:
    st.session_state.labelled_tree_data = pd.DataFrame({
        'Hierarchy': list(), 'Sentence': list()
    })

# To keep the prev labelled rows to prevent updates when current label is changed
if 'prev_labelled_rows' not in st.session_state:
    st.session_state.prev_labelled_rows = {}

# stores the test data
if 'test_data' not in st.session_state:
    st.session_state.test_data = list()

# ---------------------------------------------- The Interface

# cola - set parameters and input data
# colb - labelling and rest of workflow
cola, colb = st.columns([0.8,1])

# ---------------------------------------------- Model Fine-Tuning

with colb:

    st.title('🦾 Train Model')
    st.write('🔥 Fine-tune a model with the labelled data.')

    # split into 2 columns
    colt1, colt2 = st.columns([1.5, 1])

    with colt1:
        # upload the dataset
        up_test_data = st.file_uploader("👨🏻‍💻 Upload Test Dataset File", type={"csv", "txt"})

    if up_test_data is not None:

        # read csv dataset file
        st.session_state.test_data = pd.read_csv(up_test_data)

        # select the text column of dataset to label
        with colt2:
            st.session_state.test_text_column_name = st.selectbox(
                '💬 Select Text Column',
                st.session_state.test_data.columns
                )
        
            # select the ground truth label column of the dataset
            st.session_state.test_label_column_name = st.selectbox(
                '✏️ Select Label Column',
                [colname for colname in st.session_state.test_data.columns 
                if colname != st.session_state.test_text_column_name]
                )
    
    # enter label name
    st_tags(['prajjwal1/bert-tiny'], text='Enter to Add More', label='🤖 Enter HF Model Names', key='classifier_names')
    

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
        col1, col2, col3 = st.columns([2, 1, 1])

        # select the column of dataset to label
        with col1:
            st.session_state.text_column_name = st.selectbox(
                '💬 Select Text Column of the Dataset',
                st.session_state.data.columns
                )
        
        with col2:

            st.write('')
            st.write('')
            # start extracting the embeddings from the data for labelling ease
            st.button('🪄 Prepare Data', use_container_width=True, key='prep_data_button')
        
        with col3:

            st.write('')
            st.write('')
            # the cluster data button
            # if st.session_state.sent_emb_extractor.embeddings is not None:
            st.button('💡 Cluster Data', use_container_width=True, key='clust_data_button', 
                      disabled=(not st.session_state.prep_data_button) and (st.session_state.sent_emb_extractor.embeddings is None))

        col5, col6 = st.columns([1.5, 1.5])
        
        with col5:

            # ---------------------------------------------- Customization to the Sentence Embedding Model

            # Contains all the customization option for the Sentence Transformer
            customize_sent_emb_modal = st.expander('⚙️ Sentence Embedding Customization')

            # add all the customization UI
            sentence_embedding_customizer(customize_sent_emb_modal)

        with col6:

            # ---------------------------------------------- Customization to the Sentence Embedding Clustering Model

            # Contains all the customization option for the Clustering Strategy
            customize_emb_clust_modal = st.expander('⚙️ Embedding Clustering Customization')

            # add all the customization UI
            embedding_clustering_customizer(customize_emb_clust_modal)

        col7, col8 = st.columns([3, 1])

        with col7:

            # enter label name
            st_tags(['Negative', 'Neutral', 'Positive'], text='Enter to Add More', label='🔖 Enter Label Name', key='label_names')
        
        with col8:

            st.write('')
            # select label name
            st.selectbox('✏️ Current Label', st.session_state.label_names, help='Choose the Label which you are currently labelling', key='current_label')


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

with colb:

    if st.session_state.clust_data_button and (st.session_state.sent_emb_extractor.embeddings is not None):

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

    # st.header('🦾 Unlabelled Data')
    # st.write('🕹 Choose `Current Label` and use checkboxes to label the following data.')

    # display tree dataframe
    tree_dataframe = nested_dataframe_customizer(st.session_state.sent_emb_clusterer.tree_dataframe, 'original_data_tree', height=970)

with cola:

    # get the selected rows which are basically the just labelled rows
    current_labelled_rows = tree_dataframe['selected_rows']

    # Only run if the labelled data hasn't already been updated
    if len(current_labelled_rows) \
        and (not all((st.session_state.prev_labelled_rows.get(k) == v for k, v in current_labelled_rows[0]['_selectedRowNodeInfo'].items()))):
        
        # store the previous tree dataframe which was added to the labelled data
        st.session_state.prev_labelled_rows = current_labelled_rows[0]['_selectedRowNodeInfo']

        # Remove the selected row node info and add the `current label`
        for row in current_labelled_rows:
            del row['_selectedRowNodeInfo']
            row['Hierarchy'] = f"{st.session_state.current_label}/{row['Hierarchy']}"

        # st.dataframe(labelled_rows)

        if len(current_labelled_rows):
            # updating the labelled dataset by adding newly labelled sentences
            st.session_state.labelled_tree_data = pd.concat([st.session_state.labelled_tree_data, pd.DataFrame(current_labelled_rows)])

            # st.dataframe(st.session_state.labelled_tree_data)

    st.header('📝 Labelled Data')
    st.write('📛 View Labelled Data.')

    # updating the tree dataframe displaying labelled data
    st.session_state.labelled_tree_dataframe = nested_dataframe_customizer(st.session_state.labelled_tree_data, 'labelled_data_tree', False, True, 0, height=len(st.session_state.label_names) * 100)

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