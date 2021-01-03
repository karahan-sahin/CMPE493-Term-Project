from preprocessing import *

def execute_preprocessing():
    _corpora, _dictionary = metadata_extractor()
    _TF_IDF_Vectors = tf_idf_calculator(_corpora, _dictionary)
    _TF_IDF_Matrix = tf_idf_matrix_calculator(_TF_IDF_Vectors, _corpora, _dictionary)
    _odd_query_vector = query_analyzer(_dictionary, False)
    _relevance_analysis = relevance_analyzer(_odd_query_vector, _corpora, _TF_IDF_Matrix)

    return _relevance_analysis

def first_K_batch(batch_size=None):
    if not batch_size:
        return

    