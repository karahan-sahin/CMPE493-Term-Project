import json
from preprocessing import *

def execute_preprocessing():
    _corpora, _dictionary = metadata_extractor()
    print("Metadata extraction completed.")
    _TF_IDF_Vectors = tf_idf_calculator(_corpora, _dictionary)
    print("TF-IDF-Vectors have been created.")
    _TF_IDF_Matrix = tf_idf_matrix_calculator(_TF_IDF_Vectors, _corpora, _dictionary)
    print("TF-IDF-Matrix have been finalized.")
    _odd_query_vector = query_analyzer(_dictionary, False)
    print("Odd-numbered-queries have been analyzed.")
    _relevance_analysis = relevance_analyzer(_odd_query_vector, _corpora, _TF_IDF_Matrix)
    print("Relevance analysis have been done with odd-numbered-queries.")

    return _relevance_analysis

def first_K_batch(batch_size=None):
    if not batch_size:
        return

    _relevance_analysis = execute_preprocessing()
    fout = open("results.json", "w", encoding="utf-8")
    json.dump(_relevance_analysis, fout)
    fout.flush()
    fout.close()

if __name__ == "__main__":
    first_K_batch(1)