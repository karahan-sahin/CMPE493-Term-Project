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

    return corpora, dictionary

def dict_dump(dictionary, name):
    fout = open(f"{name}.json","w",encoding="utf-8")
    json.dump(dictionary, fout)
    fout.flush()
    fout.close()

def tf_idf_calculator(corpora, dictionary): 
    for count_dicts in corpora.values():
        for word, freq in count_dicts.items():

            # Fetch tf and idf
            term_frequency = freq
            inverted_doc_freq = math.log10(len(corpora.keys())/dictionary[word])
            count_dicts[word] = term_frequency * inverted_doc_freq

    return corpora

def query_analyzer(dictionary, isEven):
    # A dictionary for query vectors
    # topicID: query_vector  
    query_dicts = defaultdict(dict)

    for topic in match.topic_extractor(isEven=isEven):

        # tokenize query 
        tokens = tokenize(topic["query"])

        # create a count_dict
        count_dicts = defaultdict(int)
        for token in tokens:
            count_dicts[token] += 1

        # Create a query vector with tfidf weighting
        for word, freq in count_dicts.items():

            term_frequency = freq
            inverted_doc_freq = math.log10(len(dictionary.keys())/dictionary[word])

            count_dicts[word] = term_frequency * inverted_doc_freq
        query_dicts[topic["topic_id"]] = count_dicts

    return query_dicts

def relevance_analyzer(query_dicts, corpora):
    result_data = defaultdict(dict)
    # For each odd numbered topic 
    # topic: odd topic ID
    for topic, q_dict in query_dicts.items():

        # Scores dict to be sorted 
        # docID: score
        scores = dict.fromkeys(list(corpora.keys()), 0)

        # Normalizing QUERY to a binary vector
        total = 0
        for wval in q_dict.values():
            total += wval**2

        qvec_norm = total**0.5

        for docID in list(corpora.keys()):
            
            doc_dict = corpora[docID]

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

        fout = open(f"tra/{topic}.json", "w", encoding="utf-8")
        json.dump(sorted_by_scores, fout)
        fout.flush()
        fout.close()

    return result_data

