import csv
import json
import math
import pandas as pd

import match
import string
import nltk
from nltk.corpus import stopwords

from collections import defaultdict


# For all docs  -->  docID: count_dict

# For all word --->  word: collection frequency

# Remove punctuations

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

            # Add to corpora
            corpora[cord_uid] = counts

            # Remove repeated docs
            # Maybe outdated docs can be eliminated using upload_date from metacsv? 
            if corpora[cord_uid]:
                if len(corpora[cord_uid]) > len(counts):
                    continue
            else:
                corpora[cord_uid] = counts

    return corpora, dictionary

def dict_dump(dictionary):
    with open("dictionary.json","w",encoding="utf-8") as f:
        json.dump(dictionary,f, indent= 2, ensure_ascii = False)

def tf_idf_calculator(corpora, dictionary):
    # arrays for 
    TF_IDF_vectors = []

    for counts in corpora.values():

        # Create count vector from count_dicts
        count_vector = []

        # Iterate over dictionary for each word
        for word, freq in dictionary.items():

            # Fetch tf and idf
            term_frequency = counts[word]
            if term_frequency == 0:
                count_vector.append(0)
                continue

            inverted_doc_freq = math.log10(len(corpora.keys())/freq)

            0# If the term doesn't exist
            # then tf = 
            count_vector.append(term_frequency * inverted_doc_freq)

        # Append vectors for the 2D array
        TF_IDF_vectors.append(count_vector)

    return TF_IDF_vectors

def tf_idf_matrix_calculator(TF_IDF_vectors, corpora, dictionary): 
    tf_idf_matrix = pd.DataFrame(TF_IDF_vectors, index=list(corpora.keys()), columns=dictionary.keys())

    # pickled tf_idf matrix just in case
    tf_idf_pickle = tf_idf_matrix.to_pickle("./tf_idf.pickle")

    return tf_idf_matrix

def query_analyzer(dictionary, isEven):
    # A dictionary for query vectors
    # topicID: query_vector  
    query_vectors = defaultdict(list)

    for topic in match.topic_extractor(isEven=isEven):

        query_vector = []

        # tokenize query 
        tokens = tokenize(topic["query"])

        # create a count_dict
        counts = defaultdict(int)
        for token in tokens:
            counts[token] += 1

        # Create a query vector with tfidf weighting
        for word, freq in dictionary.items():

            term_frequency = counts[word]
            if term_frequency == 0:
                query_vector.append(0)
                continue

            inverted_doc_freq = math.log10(len(dictionary.keys())/freq)
            query_vector.append(term_frequency * inverted_doc_freq)
        query_vectors[topic["topic_id"]] = query_vector

    return query_vectors

def relevance_analyzer(query_vectors, corpora, tf_idf_matrix):
    result_data = defaultdict(dict)
    # For each odd numbered topic 
    # topic: odd topic ID
    for topic, q_vec in query_vectors.items():

        # Scores dict to be sorted 
        # docID: score
        scores = dict.fromkeys(list(corpora.keys()), 0)

        # Normalizing QUERY to a binary vector
        total = 0
        for word in q_vec:
            total += word**2

        qvec_norm = total**0.5

        for docID in list(corpora.keys()):
            
            doc_vec = tf_idf_matrix.loc[docID]

            # Normalizing DOCUMENT to a binary vector
            total = 0
            for word in doc_vec:
                total += word**2

            dvec_norm = total**0.5
            
            # Multiply vectors
            sums = 0
            for q, d in zip(q_vec, doc_vec):
                sums += q * d

            score = sums / (qvec_norm * dvec_norm) # qvec_norm: query vector l2 norm, dvec_norm: document vector l2 norm

            scores[docID] = score

        # Sort dictionary by values
        sorted_by_scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True))
        result_data[topic] = sorted_by_scores

    return result_data
