import csv
import json
import math

import match
import string
import nltk
from nltk.corpus import stopwords

from collections import defaultdict

# Tokenizer and Stemmer function
def tokenize(text):
    trans_table = {ord(c): None for c in string.punctuation}    
    stemmer = nltk.PorterStemmer()
    tokens = [word for word in nltk.word_tokenize(text.translate(trans_table)) if word not in set(stopwords.words('english'))] 
    stems = [stemmer.stem(item) for item in tokens]

    return stems

def metadata_extractor():
    corpora = defaultdict(dict)
    dictionary = defaultdict(int)
    # open the file and extract data
    relevancy_keys = set(match.topic_relevancy_extractor().keys())
    sum_dl = 0
    with open('metadata.csv', encoding = "utf8", errors ='replace') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
        
            # access metadata
            cord_uid = row['cord_uid']
            title = row['title']
            abstract = row['abstract']

            if cord_uid not in relevancy_keys:
                continue

            # concatinate content
            content = title + " " + abstract

            # tokenize (with lowercase)
            tokens = tokenize(content.lower())

            # Find the unique set of words in document and
            # add for inverse doc freq 
            unique_tokens = set(tokens)
            for word in unique_tokens:
                dictionary[word] += 1

            # counts dictionary for the document--> word: freq
            counts = defaultdict(int)
            # This is for count vector
            for word in tokens:
                counts[word] += 1
            
            doc_length = len(tokens)
            sum_dl += doc_length

            # Add to corpora
            corpora[cord_uid] = counts
    
    avdl = sum_dl / len(corpora)

    return corpora, dictionary, avdl

def dict_dump(dictionary, name):
    fout = open(f"{name}.json","w",encoding="utf-8")
    json.dump(dictionary, fout)
    fout.flush()
    fout.close()

def tf_idf_calculator(corpora, dictionary): 

    tf_idf_dictionary = defaultdict(dict())

    for docID, count_dicts in corpora.items():
        tdf_idf_vec = defaultdict(int)
        for word, freq in count_dicts.items():

            # Fetch tf and idf
            term_frequency = freq
            inverted_doc_freq = math.log10(len(corpora.keys())/dictionary[word])
            tdf_idf_vec[word] = term_frequency * inverted_doc_freq
        
        tf_idf_dictionary[docID] = tdf_idf_vec

    return tf_idf_dictionary

def bm25_calculator(corpora, dictionary, avdl): 
    
    bm_25_dictionary= defaultdict(dict())

    k1 = (1.2 or 2)
    b = 0.75

    for docID, count_dicts in corpora.items():
        bm_25_vec = defaultdict(int)
        doc_length = sum(count_dicts.values())
        for word, freq in count_dicts.items():
            # Fetch tf and idf
            term_frequency = freq
            inverted_doc_freq = math.log10(len(corpora.keys())/dictionary[word])
            
            bm25_score = inverted_doc_freq * (((k1+1)*term_frequency) / (k1*( (1-b) + b*(doc_length/avdl)) + term_frequency) )

            bm_25_vec[word] = bm25_score

        bm_25_dictionary[docID] = bm_25_vec

    return bm_25_dictionary

def query_analyzer(dictionary, N, isEven, avdl):
    # A dictionary for query vectors
    # topicID: query_vector  
    query_tf_idf_dicts = defaultdict(dict)
    query_bm25_dicts = defaultdict(dict)

    for topic in match.topic_extractor(isEven=isEven):

        # tokenize query 
        tokens = tokenize(topic["query"])

        # create a count_dict
        count_dict = defaultdict(int)
        for token in tokens:
            count_dict[token] += 1

        k1 = (1.2 or 2)
        b = 0.75
        query_length = len(tokens)

        tf_idf_vec = defaultdict(int)
        bm25_vec = defaultdict(float)

        # Create a query vector with tfidf weighting
        for word, freq in count_dict.items():
            
            term_frequency = freq
            inverted_doc_freq = math.log10(N/dictionary[word])
            
            bm25_score = inverted_doc_freq * (((k1+1)*term_frequency) / (k1*( (1-b) + b*(query_length/avdl)) + term_frequency) )
            
            bm25_vec[word] = bm25_score
            tf_idf_vec[word] = term_frequency * inverted_doc_freq


        query_tf_idf_dicts[topic["topic_id"]] = tf_idf_vec
        query_bm25_dicts[topic["topic_id"]] = bm25_vec


    return query_tf_idf_dicts, query_bm25_dicts

def write_to_file(topic_id, sorted_relevance_dict, score_type):
    fout = open(f"tra/{score_type}_results.txt", "a", encoding="utf-8")
    for ix, (doc_id, relevance) in enumerate(sorted_relevance_dict.items()):
        if ix % 1000 == 0:
            fout.flush()
        fout.write(f"{topic_id}\tQ0\t{doc_id}\t{(ix+1)}\t{relevance}\tSTANDARD\n")
    fout.close()

def cos_sim_relevance_analyzer(query_dicts, weighted_dict, score_type):
    result_data = defaultdict(dict)
    # For each odd numbered topic 
    # topic: odd topic ID
    for topic, q_dict in query_dicts.items():

        # Scores dict to be sorted 
        # docID: score
        scores = dict.fromkeys(list(weighted_dict.keys()), 0)

        # Normalizing QUERY to a binary vector
        total = 0
        for wval in q_dict.values():
            total += wval**2

        qvec_norm = total**0.5

        for docID in list(weighted_dict.keys()):
            
            doc_dict = weighted_dict[docID]

            # Normalizing DOCUMENT to a binary vector
            total = 0
            for wval in doc_dict.values():
                total += wval**2

            dvec_norm = total**0.5
            
            # Multiply vectors
            sums = 0
            q_dict_keys = set(q_dict.keys())
            doc_dict_keys = set(doc_dict.keys())
            intersected_keys = q_dict_keys.intersection(doc_dict_keys)

            for intersect_key in list(intersected_keys):
                sums += q_dict[intersect_key] * doc_dict[intersect_key]

            score = sums / (qvec_norm * dvec_norm) # qvec_norm: query vector l2 norm, dvec_norm: document vector l2 norm

            scores[docID] = score

        # Sort dictionary by values
        sorted_by_scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True))
        result_data[topic] = sorted_by_scores

        write_to_file(topic, sorted_by_scores, score_type)

    # {topic: {docID: SCORE}}
    return result_data

def reciprocal_ranking_fusion(tf_idf_all_scores, bm25_all_scores):    

    result_data = defaultdict(dict())

    for topic in tf_idf_all_scores.keys():
        rrp_scores = defaultdict(float)
        tf_idf_scores = tf_idf_all_scores[topic]
        bm25_scores = bm25_all_scores[topic]
        bm25_scores_keys = list(bm25_scores.keys())

        # (rank ,docID)
        # (rank ,docID)

        # tf_idf_scores = enumerate(tf_idf_all_scores[topic].keys(), 1)
        # bm25_scores = bm25_all_scores[topic].keys(), 1)

        for ix, docID in enumerate(tf_idf_scores.keys()):

            tf_idf_rank = ix
            bm25_rank = bm25_scores_keys.index(docID)

            rrp_scores[docID] = (1 / tf_idf_rank+1) + (1 / bm25_rank+1)

        sorted_by_scores = dict(sorted(rrp_scores.items(), key=lambda item: item[1],reverse=True))
        result_data[topic] = sorted_by_scores

        write_to_file(topic, sorted_by_scores,"rrp")

    return result_data





    




