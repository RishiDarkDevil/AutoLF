# ---------------------------------------------- import necessary libraries

# Streamlit WebApp
import streamlit as st

# the model details
from .model_info import MODEL_PREPROCESSING

# the clustering details
from .clustering_info import CLUSTERING_FUNCTIONS, add_cluster_args_and_add_customization

# hierarchical dataframe visualization
from st_aggrid import JsCode, AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode

# ---------------------------------------------- Add Customizer Expanders

# ---------------------------------------------- Customizer for Sentence Embedding Extraction
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
        1, None, 1024, 1, '%d', 
        key='sent_emb_batch_size', 
        help='Batch Size to be used by the Sentence Transformer Model') # , disabled=st.session_state.rand

# ---------------------------------------------- Customizer for Embedding Clustering
def embedding_clustering_customizer(customize_modal):
    '''
    Adds the UI for the embedding clustering customizer to the expander `modal`
    :customize_modal: the expander modal to which the other customization options needs to be added
    '''
    # Select Clustering Strategy for Sentence Embedding
    customize_modal.selectbox(
        'Clustering Strategy',
        CLUSTERING_FUNCTIONS.keys(),
        index=0, 
        key='clust_strat_name',
        help='The Clustering Strategy for the Sentence Embeddings'
        )
    
    # updating the clustering args dictionary with the user input values
    add_cluster_args_and_add_customization(
        st.session_state.clust_strat_name, 
        customize_modal,
        st.session_state.clustering_args[st.session_state.clust_strat_name]
        )
    
# ---------------------------------------------- Customizer for Tree DataFrame
def nested_dataframe_customizer(tree_dataframe, key=None, select_to_remove=True, reload_data=False, num_expanded=-1, update_mode=GridUpdateMode.SELECTION_CHANGED, height=1000):
    '''
    Customizes and Adds UI for the Tree DataFrame
    :tree_dataframe: A dataframe with columns `Hierarchy`, `Sentence` displaying the hierarchical data
    :key: key for this component 
    :select_to_remove: whether selection of rows should remove them
    :reload_data: Whether to update the Tree Data when input dataframe is changed
    :num_expanded: How many levels to expand in the tree data [-1 indicates all levels are expanded]
    :update_mode: specific cases when to auto update the tree data
    '''
    # tree dataframe customization options
    gb = GridOptionsBuilder.from_dataframe(tree_dataframe)
    
    # # customizing the tree dataframe

    # selection of multiple rows, children, etc using checkbox
    gb.configure_selection(selection_mode='multiple', use_checkbox=True, header_checkbox=True)

    # configuring the columns with various properties
    # Hid the `Hierarchy` column
    gb.configure_column('Hierarchy', hide = 'True')
    # Add hover tooltip full sentence display on Sentence for easier viewing
    gb.configure_column('Sentence', tooltipField = 'Sentence')

    if select_to_remove:
        # row deletion on checkbox select js code
        js = JsCode("""
        function(e) {
            let api = e.api;     
            // Get the selected rows   
            var sel = api.getSelectedRows();    
            // Remove the selected rows
            api.applyTransactionAsync({remove: sel});
            sel[0].data // IDK why adding this makes the deleted data to be returned which I wanted anyways.. So not bothering to change this : /
        };
        """)
        # inject the js code in the grid
        gb.configure_grid_options(onRowSelected=js)

    # build the options to add extra features like tree data
    gridOptions = gb.build()

    # enables pagination
    gb.configure_pagination()
 
    # the grouped column customization
    gridOptions['autoGroupColumnDef']= {
            'cellRendererParams': {
                'checkbox': True,
            }
        }
    # Tree Data Handling
    gridOptions['treeData']=True
    gridOptions['animateRows']=True
    gridOptions['groupDefaultExpanded']=num_expanded
    gridOptions['getDataPath']=JsCode(''' function(data){
        return data.Hierarchy.split("/");
    }''').js_code

    # Tree DataFrame
    tree_dataframe = AgGrid(
        tree_dataframe,
        gridOptions=gridOptions,
        height=height,
        width='100%',
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        filter=True,
        update_mode=update_mode,
        theme='material',
        tree_data=True,
        key=key,
        reload_data=reload_data,
        custom_css={"#gridToolBar": {"padding-bottom": "0px !important"}}
    )

    return tree_dataframe

        