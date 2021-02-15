import re
import csv
import sys
import json
import math

import match
import string
import nltk
import contractions

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
from clustering import calc_cosine_similarity as ccs

trans_table = {ord(c): " " for c in string.punctuation}

def get_wordnet_pos(tag):
    '''
    Retrieved from: https://towardsdatascience.com/preprocessing-text-data-using-python-576206753c28
    '''
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

# Tokenizer and Lemmatizer
def tokenize(text, pos_tag = False):
    text_tokens = [contractions.fix(word) for word in text.split()]
    text = ' '.join(text_tokens)

    text = re.sub("http(s){0,1}:\/\/[\w\/.]+", " ", text) # Remove links
    text = re.sub(r"\b\d+?\b", " ", text) # Remove numbers
    text = text.encode().decode("unicode-escape", errors="ignore") # Convert unicode characters to ASCII forms
    text = re.sub(r"\W+", " ", text) # Remove non-word characters

    # Removes stopwords and punctuation from all tokens given in text
    tokens = [word for word in word_tokenize(text.translate(trans_table)) if word not in set(stopwords.words('english'))]

    # Expands contraction in all tokens
    # e.g. "John's big" ->
    # [('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ')]
    tokens = nltk.tag.pos_tag(tokens)

    # Combines the token with its tag retrieved from wordnet
    tokens = [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in tokens]
    wnl = WordNetLemmatizer() # Initializer lemmatizer

    # If pos_tag enabled, tokens will be returned with their pos_tag values
    if pos_tag:
        tokens = [(wnl.lemmatize(word, tag), tag) for word, tag in tokens]
    else:
        tokens = [wnl.lemmatize(word, tag) for word, tag in tokens]

    return tokens

def read_body(abs_path):
    '''
    -> Accumulates the body data
    which is an object list
    '''
    text = str()
    fin = open(f"{abs_path}.json", "r")
    json_obj = json.load(fin)
    body_text_list = json_obj["body_text"]
    for body_text in body_text_list:
        # Avoided table indicating body_text dicts
        if re.match("(T|t)able\W\d+", body_text["section"]) == None:
            text += body_text["text"]
    fin.close()
    return text

def metadata_extractor():
    corpora = defaultdict(dict)
    dictionary = defaultdict(int)
    # open the file and extract data
    relevancy_keys = set(match.topic_relevancy_extractor().keys())
    sum_dl = 0 # Accumulated document length
    '''
    -> If you want to include body, uncomment the line below
    '''
    # path_prefix = "./2020-07-16/document_parses/pdf_json/"
    with open('metadata.csv', encoding = "utf8", errors ='replace') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            # access metadata
            cord_uid = row['cord_uid']
            title = row['title']
            abstract = row['abstract']
            # sha = row['sha']

            if cord_uid not in relevancy_keys:
                continue

            '''
            -> If you want to include body, uncomment the lines below
            '''
            # try:
            #     body = read_body(str(path_prefix + sha))
            #     # concatenate content
            #     content = title + " " + abstract + " " + body
            # except BaseException:
            #     # concatenate content
            #     content = title + " " + abstract

            content = title + " " + abstract

            # tokenize (with lowercase)
            tokens = tokenize(content.lower())

            # Find the unique set of words in document and
            # add for inverse doc freq
            unique_tokens = set(tokens)
            for word in unique_tokens:
                dictionary[word] += 1

            # counts dictionary for the document --> word: freq
            counts = defaultdict(int)
            # This is for count vector
            for word in tokens:
                counts[word] += 1

            doc_length = len(tokens)
            sum_dl += doc_length

            # Add to corpora
            corpora[cord_uid] = counts

    avdl = sum_dl / len(corpora) # Calculates average document length

    return corpora, dictionary, avdl

def dict_dump(dictionary, name):
    '''
    -> Outputs the dictionary to a json file
    '''
    fout = open(f"{name}.json","w",encoding="utf-8")
    json.dump(dictionary, fout)
    fout.flush()
    fout.close()

def tf_idf_calculator(corpora, dictionary):
    '''
    -> TF-IDF value calculator
    '''
    tf_idf_dictionary = defaultdict(dict)
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
    bm_25_dictionary= defaultdict(dict)

    '''
    -> Selected the BM25 parameters as below 
    '''
    k1 = 1.2
    b = 0.75

    for docID, count_dicts in corpora.items():
        bm_25_vec = defaultdict(int)
        doc_length = sum(count_dicts.values())
        for word, freq in count_dicts.items():

            # Fetch tf and idf
            term_frequency = freq
            inverted_doc_freq = math.log10(len(corpora.keys())/dictionary[word])
            # BM25 score calculation
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

        # tokenize query, ADDING NARRATIVES DRASTICALLY DROPPED THE RESULTS
        tokens_with_tag = tokenize(topic["query"].lower(), pos_tag = True)
        tokens = []

        '''
        -> The section below handles the query expansion
        which led our results to a decrease

        print(syns[0].name()) --> plan.n.01
        print(syns[0].lemmas()[0].name())  --> plan
        '''

        for token_with_tag in tokens_with_tag:
            token = token_with_tag[0]
            tag = token_with_tag[1]
            list_token_with_tag = list(token_with_tag)
            list_token_with_tag.append("01")
            token_with_dot = ".".join(list_token_with_tag)

            for syn in wn.synsets(token, tag):
                syn_with_tag = syn.name()
                if syn_with_tag.split(".")[0] == token:
                    continue
                w1 = wn.synset(token_with_dot)
                w2 = wn.synset(syn_with_tag)
                try:
                    if w1.wup_similarity(w2) > 0.7:
                        try:
                            accessible_syn = syn.lemmas()[0].name()
                            accessed_syn = dictionary[accessible_syn]
                            tokens.append(accessible_syn)
                        except KeyError:
                            pass
                except BaseException:
                    pass

            tokens.append(token)

        # create a count_dict
        count_dict = defaultdict(int)
        for token in tokens:
            try:
                isReachable = dictionary[token]
            except KeyError:
                continue
            count_dict[token] += 1

        '''
        -> Selected the BM25 parameters as below 
        '''
        k1 = 1.2
        b = 0.75
        query_length = len(tokens)

        tf_idf_vec = defaultdict(int)
        bm25_vec = defaultdict(float)

        # Create a query vector with tfidf weighting
        for word, freq in count_dict.items():

            term_frequency = freq
            # If dictionary does not contain a word in a query,
            # we accept its count as 1.
            try:
                inverted_doc_freq = math.log10(N/dictionary[word])
            except ZeroDivisionError:
                inverted_doc_freq = math.log10(N/(dictionary[word]+1))

            # BM25 score calculation
            bm25_score = inverted_doc_freq * (((k1+1)*term_frequency) / (k1*( (1-b) + b*(query_length/avdl)) + term_frequency))

            bm25_vec[word] = bm25_score
            tf_idf_vec[word] = term_frequency * inverted_doc_freq

        query_tf_idf_dicts[topic["topic_id"]] = tf_idf_vec
        query_bm25_dicts[topic["topic_id"]] = bm25_vec

    return query_tf_idf_dicts, query_bm25_dicts

def write_to_file(topic_id, sorted_relevance_dict, score_type):
    '''
    -> Dumps the given sorted relevance dict to the related file
    in folder tra
    '''
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
            # To calculate DOT product between vectors, we
            # aim to find the intersecting keys to achieve
            # such operation
            intersected_keys = q_dict_keys.intersection(doc_dict_keys)

            for intersect_key in list(intersected_keys):
                sums += q_dict[intersect_key] * doc_dict[intersect_key]

            score = sums / (qvec_norm * dvec_norm) # qvec_norm: query vector l2 norm, dvec_norm: document vector l2 norm

            scores[docID] = score

        # Sort dictionary in descending order by values
        sorted_by_scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True))
        result_data[topic] = sorted_by_scores

        write_to_file(topic, sorted_by_scores, score_type)

    # {topic: {docID: SCORE}}
    return result_data


def get_doc_rank(result_data):
    '''
    -> Enumerates any sorted score dictionary
    to utilize indexing
    '''
    result_data_with_rank = defaultdict(dict)
    for topic, sorted_by_scores in result_data.items():
        doc_rank_dict = defaultdict(int)
        ix = 1
        for docID, score in sorted_by_scores.items():
            doc_rank_dict[docID] = (ix, score)
            ix += 1

        result_data_with_rank[topic] = doc_rank_dict
    
    return result_data_with_rank

def reciprocal_ranking_fusion(tf_idf_all_scores, bm25_all_scores, score_type):    

    result_data = defaultdict(dict)

    # Append rankings to the sorted score dictionaries
    tf_idf_with_rank = get_doc_rank(tf_idf_all_scores)
    bm25_with_rank = get_doc_rank(bm25_all_scores)

    # We've utilized our k value in RRF calculation
    # by trying k=1 to k=80, 10 incrementation at each time
    # and stored our results in the data sheet we've
    # provided in our presentation. You can control the
    # results and select the one that suits you.
    K = 1

    for topic in tf_idf_with_rank.keys():

        rrf_scores = defaultdict(float)        
        tf_idf_scores = tf_idf_with_rank[topic]
        bm25_scores = bm25_with_rank[topic]

        for docID in tf_idf_scores.keys():

            tf_idf_rank = tf_idf_scores[docID][0]
            # If TF-IDF score dictionary does not
            # contain the same document for BM_25
            # dictionary in clustering mode,
            # it assumes its rank as the last one in the
            # BM25 score dictionary
            try:
                bm25_rank = bm25_scores[docID][0]
            except:
                bm25_rank = len(tf_idf_scores)

            rrf_scores[docID] = (1 / (tf_idf_rank + K)) + (1 / (bm25_rank + K))

        # Sorted score dictionary in descending order by values
        sorted_by_scores = dict(sorted(rrf_scores.items(), key=lambda item: item[1],reverse=True))
        result_data[topic] = sorted_by_scores

        write_to_file(topic, sorted_by_scores, score_type)

    return result_data

def evaluate_with_related_clusters(_clusters, _cluster_doc_map, _query_dicts, _weighted_dictionary, score_type):
    '''
    -> It utilizes the provided clusters for the query
    and calculates the scores based on the documents
    that are in the provided clusters
    '''
    result_data = defaultdict(dict)
    for topicID, query_vec in _query_dicts.items():
        query_related_docs = list()
        cluster_scores = dict()

        for i in range(len(_clusters)):
            cluster_result = ccs(_clusters[str(i)], query_vec)
            cluster_scores[i] = cluster_result

        # Scores of the each cluster,
        # based on the cosine similarity
        # of its centroid and the given query
        cluster_scores = dict(sorted(cluster_scores.items(), key=lambda item: item[1],reverse=True))
        preferences = list(cluster_scores.keys())

        # We've selected the most related 5 clusters
        # to evaluate the documents, which we deem as
        # related with the given query.
        for cluster in preferences[:5]:
            query_related_docs += _cluster_doc_map[str(cluster)]

        scores = dict.fromkeys(query_related_docs, 0)

        # Normalizing QUERY to a binary vector
        total = 0
        for wval in query_vec.values():
            total += wval**2

        qvec_norm = total**0.5

        for docID in query_related_docs:

            doc_dict = _weighted_dictionary[docID]

            # Normalizing DOCUMENT to a binary vector
            total = 0
            for wval in doc_dict.values():
                total += wval**2

            dvec_norm = total**0.5

            # Multiply vectors
            sums = 0
            q_dict_keys = set(query_vec.keys())
            doc_dict_keys = set(doc_dict.keys())

            # To calculate DOT product between vectors, we
            # aim to find the intersecting keys to achieve
            # such operation
            intersected_keys = q_dict_keys.intersection(doc_dict_keys)

            for intersect_key in list(intersected_keys):
                sums += query_vec[intersect_key] * doc_dict[intersect_key]

            score = sums / (qvec_norm * dvec_norm) # qvec_norm: query vector l2 norm, dvec_norm: document vector l2 norm

            scores[docID] = score

        # Sort score dictionary in descending order by values
        sorted_by_scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True))
        result_data[topicID] = sorted_by_scores

        write_to_file(topicID, sorted_by_scores, score_type)

    return result_data
    
def score_combination(tf_idf_all_scores, bm25_all_scores, score_type = ""):
    '''
    -> We've decided to utilize more ranking functions, thus
    we've crawled several articles and selected 3 more ranking
    functions from here:
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.83.7587&rep=rep1&type=pdf
    '''

    # 1	Q0	pl48ev5o	1	0.5970826005488854	STANDARD
    # 1	Q0	xwi9pdd2	2	0.5970826005488854	STANDARD 

    # 2	Q0	pl48ev5o	1	0.5970826005488854	STANDARD
    # 2	Q0	xwi9pdd2	2	0.5970826005488854	STANDARD 

    # Append rankings to the sorted score dictionaries
    tf_idf_all_scores_with_rank = get_doc_rank(tf_idf_all_scores)
    bm25_all_scores_with_rank = get_doc_rank(bm25_all_scores)

    result_data_SUM = defaultdict(dict)
    result_data_ANZ = defaultdict(dict)
    result_data_MNZ = defaultdict(dict)

    for topic in tf_idf_all_scores_with_rank.keys():
        scores_SUM = defaultdict(float)
        scores_ANZ = defaultdict(float)
        scores_MNZ = defaultdict(float)

        # Topic related scores
        tf_idf_scores_with_rank = tf_idf_all_scores_with_rank[topic]
        bm25_scores_with_rank = bm25_all_scores_with_rank[topic]

        for docID in tf_idf_scores_with_rank.keys():
            tf_idf_score = tf_idf_scores_with_rank[docID][1]

            try:
                bm25_score = bm25_scores_with_rank[docID][1]
            except:
                bm25_score = 0

            scores_SUM[docID] = tf_idf_score + bm25_score

            anz_count = 0
            tf_idf_rank = tf_idf_scores_with_rank[docID][0]
            try:
                bm25_rank = bm25_scores_with_rank[docID][0]
            except:
                bm25_rank = len(bm25_scores_with_rank)

            # MNZ or ANZ scoring acts on whether or not
            # a document lies on the first 1000 batch
            scores_MNZ[docID] = 1
            if tf_idf_rank < 1000:
                anz_count += 1
                scores_MNZ[docID] *= tf_idf_score
            if bm25_rank < 1000:
                anz_count += 1
                scores_MNZ[docID] *= bm25_score

            if scores_MNZ[docID] == 1:
                scores_MNZ[docID] = 0

            # General SUM averaging
            try:
                scores_ANZ[docID] = scores_SUM[docID] / anz_count
            except:
                scores_ANZ[docID] = 0

        result_data_SUM[topic] = scores_SUM
        result_data_ANZ[topic] = scores_ANZ
        result_data_MNZ[topic] = scores_MNZ

        # Sorted score dictionary in descending order by values
        sorted_by_scores = dict(sorted(scores_SUM.items(), key=lambda item: item[1],reverse=True))
        write_to_file(topic, sorted_by_scores, f"sum{score_type}")

        # Sorted score dictionary in descending order by values
        sorted_by_scores = dict(sorted(scores_ANZ.items(), key=lambda item: item[1],reverse=True))
        write_to_file(topic, sorted_by_scores, f"anz{score_type}")

        # Sorted score dictionary in descending order by values
        sorted_by_scores = dict(sorted(scores_MNZ.items(), key=lambda item: item[1],reverse=True))
        write_to_file(topic, sorted_by_scores, f"mnz{score_type}")
      
    return result_data_SUM, result_data_ANZ, result_data_MNZ
