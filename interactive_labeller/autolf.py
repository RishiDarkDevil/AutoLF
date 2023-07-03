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

# clustering infos
from cluster_sentence.clustering_info import CLUSTERING_FUNCTIONS

# all the customizers
from cluster_sentence.customization_utils import sentence_embedding_customizer, embedding_clustering_customizer

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

# stores the clustering arguments
if 'clustering_args' not in st.session_state:
    st.session_state.clustering_args = {
        cluster_strat: {} for cluster_strat in CLUSTERING_FUNCTIONS.keys()
    }

# prepare data button state
prep_data_button = False

# cluster data button state
clust_data_button = False

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
            prep_data_button = st.button('🪄 Prepare Data', use_container_width=True)
        
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
    if prep_data_button:
        
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
    if st.session_state.sent_emb_extractor.embeddings is not None:
        with col6:
            clust_data_button = st.button('💡 Cluster Data', use_container_width=True)

    # markmap text
    md = '''
    - 19998.0
	 - 19194.0
		 - ok sir so for the "xbox" it has better ergonomics on its controller but the "sony" "psp" has a better graphics all in all
		 - i'm doing good too i'm looking for information about video game video game consoles
	 - 19997.0
		 - 19994.0
			 - okay and are there any limitations or exclusions on compression stockings or any lower aah compression garments low extremity compression garments
			 - 19989.0
				 - it is collapsible but i don't think it's collapsible enough to fit any of our suitcase
				 - 19981.0
					 - 19344.0
						 - does it have any measurement issues or just i need to keep the thing okay
						 - so maximum accepted dimensions it's forty inches in height
					 - 19907.0
						 - 19488.0
							 - 17987.0
								 - it's fine ma'am if it's already been opened but i need to ask you to reseal it first with some tape before sending it back to our offices
								 - will be safe sealed so you don't have to worry on it
								 - it's fine ma'am if it's already been opened but i need to ask you to reseal it first with some tape before sending it back to our offices
								 - will be safe sealed so you don't have to worry on it
							 - 19336.0
								 - can you put you said it store and "colt" okay have you discussed this issue with you doctor at all
								 - it's separate from the ammunition is that still okay to check in luggage as long as it's locked in a box
						 - 19687.0
							 - 19358.0
								 - everything that i need to put of anything of liquids has to go into the the little clear zip locks bag
								 - 19030.0
									 - 17132.0
										 - it was the the seat selection and the checked bags and stuff that i saved
										 - liked checked bags carry on bags and seat selections as you choose
										 - it was the the seat selection and the checked bags and stuff that i saved
										 - liked checked bags carry on bags and seat selections as you choose
									 - 18570.0
    '''
    markmap(md,height=600)
