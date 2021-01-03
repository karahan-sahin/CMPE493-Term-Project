import json
import copy

from numpy.core.numeric import True_
from preprocessing import *

def execute_preprocessing():
    _corpora, _dictionary = metadata_extractor()
    print("Metadata extraction completed.")
    dict_dump(_corpora, "corpora")
    print("Backup corpora dumped.")
    dict_dump(_dictionary, "dictionary")
    print("Backup dictionary dumped.")
    _corpora = tf_idf_calculator(_corpora, _dictionary)
    print("TF-IDF values calculated.")
    _odd_query_vector = query_analyzer(_dictionary, False)
    print("Odd-numbered-queries have been analyzed.")
    _relevance_analysis = relevance_analyzer(_odd_query_vector, _corpora)
    print("Relevance analysis have been done with odd-numbered-queries.")

    return _relevance_analysis

def load_continue_preprocessing(loadCorpora=True, loadDict=True):
    print("Loading sequence initiated.")
    if not loadCorpora and not loadDict:
        _corpora, _dictionary = metadata_extractor()
    else:
        fin_corpora = open("corpora.json", "r", encoding="utf-8")
        _corpora = json.load(fin_corpora)
        fin_corpora.close()
        fin_dictionary = open("dictionary.json", "r", encoding="utf-8")
        _dictionary = json.load(fin_dictionary)
        fin_dictionary.close()
    print("Loading sequence completed.")

    _corpora = tf_idf_calculator(_corpora, _dictionary)
    _odd_query_vector = query_analyzer(_dictionary, False)
    _relevance_analysis = relevance_analyzer(_odd_query_vector, _corpora)
    return _relevance_analysis

def first_K_batch(batch_size=None, reload=False):
    if not batch_size:
        return

    if not reload:
        _relevance_analysis = execute_preprocessing()
    else:
        _relevance_analysis = load_continue_preprocessing()
    print("Relevance analysis completed.")

if __name__ == "__main__":
    first_K_batch(1, True)
