# ---------------------------------------------- import necessary libraries

# model building
import torch

# sentence transformer model
from sentence_transformers import SentenceTransformer

# community clustering
from sentence_transformers import util

# ---------------------------------------------- Model's Preprocessing Dictionary

# it should be key, value pair:
# key is the name of the model
# value is the dictionary of backbone name and
# function that takes in a list of sentences and preprocesses them to be input to the model
MODEL_PREPROCESSING = {
    'Sentence Transformer': {
        backbone_name: lambda inpt: inpt
        for backbone_name in [
            'all-MiniLM-L6-v2',
            'multi-qa-mpnet-base-dot-v1',
            'all-mpnet-base-v2',
            'multi-qa-distilbert-cos-v1',
            'multi-qa-MiniLM-L6-cos-v1',
            'all-distilroberta-v1',
            'all-MiniLM-L12-v2',
            'paraphrase-multilingual-mpnet-base-v2',
            'paraphrase-albert-small-v2',
            'paraphrase-MiniLM-L3-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'distiluse-base-multilingual-cased-v1',
            'distiluse-base-multilingual-cased-v2'
            ]
    }
}

# ---------------------------------------------- Model's Build Function Dictionary

# model build functions
MODEL_BUILD_FUNCTIONS = {
    'Sentence Transformer': {
        backbone_name: {
            'build_function': SentenceTransformer,
            'args': {
                'model_name_or_path': backbone_name, 
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            }
        for backbone_name in [
            'all-MiniLM-L6-v2',
            'multi-qa-mpnet-base-dot-v1',
            'all-mpnet-base-v2',
            'multi-qa-distilbert-cos-v1',
            'multi-qa-MiniLM-L6-cos-v1',
            'all-distilroberta-v1',
            'all-MiniLM-L12-v2',
            'paraphrase-multilingual-mpnet-base-v2',
            'paraphrase-albert-small-v2',
            'paraphrase-MiniLM-L3-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'distiluse-base-multilingual-cased-v1',
            'distiluse-base-multilingual-cased-v2'
            ]
    },

}

# ---------------------------------------------- Model's Post-Processing Function Dictionary

# model post processing of output
MODEL_POSTPROCESSING = {
    'Sentence Transformer': {
        backbone_name: lambda output: output
        for backbone_name in [
            'all-MiniLM-L6-v2',
            'multi-qa-mpnet-base-dot-v1',
            'all-mpnet-base-v2',
            'multi-qa-distilbert-cos-v1',
            'multi-qa-MiniLM-L6-cos-v1',
            'all-distilroberta-v1',
            'all-MiniLM-L12-v2',
            'paraphrase-multilingual-mpnet-base-v2',
            'paraphrase-albert-small-v2',
            'paraphrase-MiniLM-L3-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'distiluse-base-multilingual-cased-v1',
            'distiluse-base-multilingual-cased-v2'
            ]
    }
}