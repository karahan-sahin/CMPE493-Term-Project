import json

from preprocessing import *
from clustering import *

def execute_preprocessing():
    _corpora, _dictionary, _avdl = metadata_extractor()
    print("Metadata extraction completed.")
    dict_dump(_corpora, "corpora")
    print("Backup corpora dumped.")
    dict_dump(_dictionary, "dictionary")
    print("Backup dictionary dumped.")
    dict_dump({"avdl": _avdl}, "avdl")
    print("Backup avdl dumped.")

    _tf_idf_dictionary = tf_idf_calculator(_corpora, _dictionary)
    print("TF-IDF values calculated.")    
    dict_dump(_tf_idf_dictionary, "tf_idf")
    print("Backup TF-IDF dumped.")

    _tf_idf_clusters, _tf_idf_cluster_doc_map = compose_clusters(_tf_idf_dictionary)
    print("Clustering of documents based on TF-IDF values completed.")
    _clusters_dump_obj = { "clusters": _tf_idf_clusters, "cluster_doc_map": _tf_idf_cluster_doc_map }
    dict_dump(_clusters_dump_obj, "clusters_dump_obj_tf_idf")
    print("Backup TF-IDF Clusters dumped.")

    _bm25_dictionary = bm25_calculator(_corpora, _dictionary, _avdl)
    print("BM25 values calculated.")
    dict_dump(_bm25_dictionary, "bm25")
    print("Backup BM25 dumped.")

    _bm25_clusters, _bm25_cluster_doc_map = compose_clusters(_bm25_dictionary)
    print("Clustering of documents based on BM25 values completed.")
    _clusters_dump_obj = { "clusters": _bm25_clusters, "cluster_doc_map": _bm25_cluster_doc_map }
    dict_dump(_clusters_dump_obj, "clusters_dump_obj_bm_25")
    print("Backup BM25 Clusters dumped.")

    _query_tf_idf_dicts, _query_bm25_dicts  = query_analyzer(_dictionary, len(_corpora), False, _avdl)
    print("Odd-numbered-queries have been analyzed.")
    
    _tf_idf_clustered_relevance_analysis = evaluate_with_related_clusters(_tf_idf_clusters, _tf_idf_cluster_doc_map, _query_tf_idf_dicts, _tf_idf_dictionary, "tf_idf_with_clustering")
    _bm25_clustered_relevance_analysis = evaluate_with_related_clusters(_bm25_clusters, _bm25_cluster_doc_map, _query_bm25_dicts, _bm25_dictionary, "bm25_with_clustering")
    _rrf_clustered_relevance_analysis = reciprocal_ranking_fusion(_tf_idf_clustered_relevance_analysis, _bm25_clustered_relevance_analysis, "rrf_with_clustering")

    
    _tf_idf_relevance_analysis = cos_sim_relevance_analyzer(_query_tf_idf_dicts, _tf_idf_dictionary, "tf_idf")
    _bm25_relevance_analysis = cos_sim_relevance_analyzer(_query_bm25_dicts, _bm25_dictionary, "bm25")
    _rrf_relevance_analysis = reciprocal_ranking_fusion(_tf_idf_relevance_analysis, _bm25_relevance_analysis, "rrf")

    print("TF-IDF Relevance analysis have been done with odd-numbered-queries.")

    return _tf_idf_relevance_analysis, _bm25_relevance_analysis, _rrf_relevance_analysis, _tf_idf_clustered_relevance_analysis, _bm25_clustered_relevance_analysis, _rrf_clustered_relevance_analysis

def load_continue_preprocessing(loadCorpora=True, loadDict=True):
    print("Loading sequence initiated.")
    if not loadCorpora and not loadDict:
        _tf_idf_relevance_analysis, _bm25_relevance_analysis, _rrf_relevance_analysis = execute_preprocessing()
    else:
        fin_corpora = open("corpora.json", "r", encoding="utf-8")
        _corpora = json.load(fin_corpora)
        fin_corpora.close()

        fin_dictionary = open("dictionary.json", "r", encoding="utf-8")
        _dictionary = json.load(fin_dictionary)
        fin_dictionary.close()

        fin_avdl = open("avdl.json", "r", encoding="utf-8")
        _avdl_dictionary = json.load(fin_avdl)
        _avdl = _avdl_dictionary["avdl"]
        fin_avdl.close()

        fin_tf_idf = open("tf_idf.json", "r", encoding="utf-8")
        _tf_idf_dictionary = json.load(fin_tf_idf)
        fin_tf_idf.close()

        fin_bm25 = open("bm25.json", "r", encoding="utf-8")
        _bm25_dictionary = json.load(fin_bm25)
        fin_bm25.close()

        fin_tf_idf_cluster = open("clusters_dump_obj_tf_idf.json", "r", encoding="utf-8")
        tf_idf_cluster_dump_obj = json.load(fin_tf_idf_cluster)
        fin_tf_idf_cluster.close()
        _tf_idf_clusters = tf_idf_cluster_dump_obj["clusters"]
        _tf_idf_cluster_doc_map = tf_idf_cluster_dump_obj["cluster_doc_map"]

        fin_bm25_cluster = open("clusters_dump_obj_bm_25.json", "r", encoding="utf-8")
        bm_25_cluster_dump_obj = json.load(fin_bm25_cluster)
        fin_bm25_cluster.close()
        _bm25_clusters = bm_25_cluster_dump_obj["clusters"]
        _bm25_cluster_doc_map = bm_25_cluster_dump_obj["cluster_doc_map"]

        print("Loading sequence completed.")

        _query_tf_idf_dicts, _query_bm25_dicts  = query_analyzer(_dictionary, len(_corpora) ,False, _avdl)

        _tf_idf_clustered_relevance_analysis = evaluate_with_related_clusters(_tf_idf_clusters, _tf_idf_cluster_doc_map, _query_tf_idf_dicts, _tf_idf_dictionary, "tf_idf_with_clustering")
        _bm25_clustered_relevance_analysis = evaluate_with_related_clusters(_bm25_clusters, _bm25_cluster_doc_map, _query_bm25_dicts, _bm25_dictionary, "bm25_with_clustering")
        _rrf_clustered_relevance_analysis = reciprocal_ranking_fusion(_tf_idf_clustered_relevance_analysis, _bm25_clustered_relevance_analysis, "rrf_with_clustering")

        _tf_idf_relevance_analysis = cos_sim_relevance_analyzer(_query_tf_idf_dicts, _tf_idf_dictionary, "tf_idf")
        _bm25_relevance_analysis = cos_sim_relevance_analyzer(_query_bm25_dicts, _bm25_dictionary, "bm25")
        _rrf_relevance_analysis = reciprocal_ranking_fusion(_tf_idf_relevance_analysis, _bm25_relevance_analysis, "rrf")

    return _tf_idf_relevance_analysis, _bm25_relevance_analysis, _rrf_relevance_analysis, _tf_idf_clustered_relevance_analysis, _bm25_clustered_relevance_analysis, _rrf_clustered_relevance_analysis

def first_K_batch(batch_size=None, reload=False):
    if not batch_size:
        return

    if not reload:
        _tf_idf_relevance_analysis, _bm25_relevance_analysis, _rrf_relevance_analysis, _tf_idf_clustered_relevance_analysis, _bm25_clustered_relevance_analysis, _rrf_clustered_relevance_analysis = execute_preprocessing()
    else:
        _tf_idf_relevance_analysis, _bm25_relevance_analysis, _rrf_relevance_analysis, _tf_idf_clustered_relevance_analysis, _bm25_clustered_relevance_analysis, _rrf_clustered_relevance_analysis = load_continue_preprocessing()

    print("Relevance analysis completed.")

if __name__ == "__main__":
    first_K_batch(1, True)
