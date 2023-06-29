# ---------------------------------------------- import necessary libraries

# To prevent warning errors in pandas DataFrames
import warnings
warnings.filterwarnings('ignore')

# general
import os

# matrix manipulation
import numpy as np

# data handling
import pandas as pd

# webapp interface
import streamlit as st

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

# ---------------------------------------------- The Interface

# cola - set parameters and input data
# colb - labelling and rest of workflow
cola, colb = st.columns([0.8,1])

with cola:

    # title and intro to the wabapp
    st.title('AutoLF🏷: Interactive Labeller')
    st.write('Create Labelled Datasets for your Machine Learning Model with Ease!')

    # upload the dataset
    up_data = st.file_uploader("Upload Dataset File", type={"csv", "txt"})
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
                'Select Text Column of the Dataset',
                st.session_state.data.columns
                )
        
        # the labels to be used
        with col2:

            # enter label name
            label_name = st.text_input('Enter Label Name', value="", placeholder = 'Click Add to Update Labels')

        with col3:
            st.write('')
            st.write('')
            # add label button
            add_label = st.button('Add', use_container_width=True)

        with col4:
            st.write('')
            st.write('')
            # reset label button
            reset_label = st.button('Reset', use_container_width=True)
            
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
