# ---------------------------------------------- import necessary libraries



# ---------------------------------------------- Embedding Extraction

class EmbeddingExtractor:

    def __init__(self, model, preprocess, postprocess):
        '''
        The embedding extractor model
        :model: the model that converts the list of sequence of tokens (or tensor) to embeddings
        :preprocess: the preprocessing function that converts a list of sentences to the form the `model` expects
        :postprocess: the post-processing pipelien that converts the model's raw output into a tensor of `number of sentences x embeddding dim`
        '''
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.embeddings = None

    def extract_embeddings(self, sentence_list):
        '''
        takes a sentence list and preprocesses them and generates embeddings
        '''

        # preprocess the sentence list
        preprocessed_data = self.preprocess(sentence_list)

        # extract embedding
        embeddings = self.model(preprocessed_data)

        # postprocess embedding
        self.embeddings = self.postprocess(embeddings)




