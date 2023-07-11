# ---------------------------------------------- import necessary libraries

# matrix manipulation
import numpy as np

# data handling
import pandas as pd

# ---------------------------------------------- Utility Functions

def create_linkage_matrix(model):
    '''
    Create linkage matrix from Agglomerative Clustering Model
    :model: fitted Agglomerative Clustering Model
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix

def create_tree_from_linkage(linked):
    '''
    Converts Linkage Matrix to D3.js type tree
    :linked: Linkage Matrix of the Agglomerative Clustering Model
    '''
    # inner func to recurvise-ly walk the linkage matrix
    def recurTree(tree):
        k = tree['name']
        ## no children for this node
        if k not in inter:
            return
        for n in inter[k]:
            ## build child nodes
            node = {
                "name": n,
                "parent": k,
                "children": []
            }
            # add to children
            tree['children'].append(node)
            # next recursion
            recurTree(node)      
    
    num_rows, _ = linked.shape
    inter = {}
    i = 0
    # loop linked matrix convert to dictionary
    for row in linked:
        i += 1
        inter[float(i + num_rows)] = [row[0],row[1]]

    # start tree data structure
    tree = {
        "name": float(i + num_rows),
        "parent": None,
        "children": []
    }
    # start recursion
    recurTree(tree);
    return tree

def tree_to_markmap(tree, num_levels, sentences):
    '''
    converts D3.js type tree to MarkMap.
    :tree: the nested dictionary which is in the form of D3.js
    :num_levels: how many levels of the tree to print
    :sentences: the list of sentences whose index is refered in the leaves of the `tree`
    '''

    def recurMarkMap(node, level):
        markmap_text = ''
        if level > num_levels:
            return ''
        if node['name'] >= len(sentences):
            markmap_text += '\t'*level + f"- {node['name']}\n"
        else:
            markmap_text += '\t'*level + f"- {sentences[int(node['name'])]}\n"
        if len(node['children']):
            for children in node['children']:
                markmap_text += recurMarkMap(children, level+1)
        return markmap_text

    # recursively create the tree
    markmap_text = recurMarkMap(tree, 0)

    return markmap_text

def tree_to_dataframe(tree, num_levels, sentences):
    '''
    converts D3.js type tree to DataFrame.
    :tree: the nested dictionary which is in the form of D3.js
    :num_levels: how many levels of the tree to print
    :sentences: the list of sentences whose index is refered in the leaves of the `tree`
    '''
    def recurNestDataFrame(node, level):

        # maintain the hierarchy
        hierarchy = str(node['name'])
        
        # create the tree dataframe
        dataframe = pd.DataFrame({
                'Hierarchy': [hierarchy], 'Sentence': ['']
            })
        
        if level > num_levels:
            return ''
        if node['name'] < len(sentences):
            dataframe['Sentence'] = sentences[int(node['name'])]
            
        if len(node['children']):
            for children in node['children']:
                child_dataframe = recurNestDataFrame(children, level+1)
                child_dataframe.Hierarchy = child_dataframe.Hierarchy.apply(
                    lambda x: f'{hierarchy}/{x}' # if x.split('/')[0] != str(node['name']) else hierarchy
                    )
                dataframe = pd.concat([dataframe, child_dataframe])
        return dataframe

    # recursively create the tree
    dataframe = recurNestDataFrame(tree, 0)

    return dataframe