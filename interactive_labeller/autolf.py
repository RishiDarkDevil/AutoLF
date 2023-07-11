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
    # import pandas as pd
    # import streamlit as st
    # from st_aggrid import JsCode, AgGrid, GridOptionsBuilder
    # from st_aggrid.shared import GridUpdateMode

    # def GetLastName(row):
    #     nsarr = row['orgHierarchy'].split('|')
    #     return(nsarr[len(nsarr)-1])

    # df=pd.DataFrame({ "orgHierarchy": ['Erica Rogers', 
    #                                 'Erica Rogers|Malcolm Barrett',
    #                                 'Erica Rogers|Malcolm Barrett|Esther Baker',
    #                                 'Erica Rogers|Malcolm Barrett|Esther Baker|Brittany Hanson',
    #                                 'Erica Rogers|Malcolm Barrett|Esther Baker|Brittany Hanson|Leah Flowers',
    #                                 'Erica Rogers|Malcolm Barrett|Esther Baker|Brittany Hanson|Tammy Sutton',
    #                                 'Erica Rogers|Malcolm Barrett|Esther Baker|Derek Paul',
    #                                 'Erica Rogers|Malcolm Barrett|Francis Strickland',
    #                                 'Erica Rogers|Malcolm Barrett|Francis Strickland|Morris Hanson',
    #                                 'Erica Rogers|Malcolm Barrett|Francis Strickland|Todd Tyler',
    #                                 'Erica Rogers|Malcolm Barrett|Francis Strickland|Bennie Wise',
    #                                 'Erica Rogers|Malcolm Barrett|Francis Strickland|Joel Cooper'],
    #                 "jobTitle": [ 'CEO', 'Exec. Vice President', 'Director of Operations', 'Fleet Coordinator', 'Parts Technician',
    #                                 'Service Technician', 'Inventory Control', 'VP Sales', 'Sales Manager', 'Sales Executive',
    #                                 'Sales Executive', 'Sales Executive' ], 
    #                 "employmentType": [ 'Permanent', 'Permanent', 'Permanent', 'Permanent', 'Contract', 'Contract', 'Permanent', 'Permanent',
    #                                     'Permanent', 'Contract', 'Contract', 'Permanent' ]}, 
    # )

    # df['Name'] = df.apply(lambda row: GetLastName(row), axis=1)
    # df.insert(0, "Name", df.pop("Name"))    # move col to 0 pstn

    # gb = GridOptionsBuilder.from_dataframe(df)
    # gb.configure_selection(selection_mode="single", use_checkbox=False)
    # gb.configure_column("orgHierarchy", hide = "True")
    # gb.configure_column("Name", hide = "True")
    # gridOptions = gb.build()

    # gridOptions["autoGroupColumnDef"]= {'cellRendererParams': {'checkbox': True }}
    # gridOptions["treeData"]=True
    # gridOptions["animateRows"]=True
    # gridOptions["groupDefaultExpanded"]= -1   # expand all
    # gridOptions["getDataPath"]=JsCode("function(data){ return data.orgHierarchy.split('|'); }").js_code

    # dta = AgGrid(df, gridOptions=gridOptions, height=350, allow_unsafe_jscode=True, enable_enterprise_modules=True,
    #             update_mode=GridUpdateMode.SELECTION_CHANGED)

    # st.write(dta['selected_rows'])

# with cola:
#     st.write(tree_dataframe['selected_rows'])