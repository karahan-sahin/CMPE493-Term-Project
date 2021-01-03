import json
import copy
from preprocessing import *

def execute_preprocessing():
    _corpora, _dictionary = metadata_extractor()
    print("Metadata extraction completed.")
    dict_dump(_corpora)
    print("Backup corpora dumped.")
    dict_dump(_dictionary)
    print("Backup dictionary dumped.")
    _corpora = tf_idf_calculator(_corpora, _dictionary)
    print("TF-IDF values calculated.")
    _odd_query_vector = query_analyzer(_dictionary, False)
    print("Odd-numbered-queries have been analyzed.")
    _relevance_analysis = relevance_analyzer(_odd_query_vector, _corpora)
    print("Relevance analysis have been done with odd-numbered-queries.")

    return _relevance_analysis

def first_K_batch(batch_size=None):
    if not batch_size:
        return

    _relevance_analysis = execute_preprocessing()

if __name__ == "__main__":
    first_K_batch(1)
