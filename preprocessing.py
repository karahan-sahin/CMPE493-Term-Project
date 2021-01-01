import csv
import json
import math
import pandas as pd

import string
import nltk
from nltk.corpus import stopwords

from collections import defaultdict

from match import odd_topics

# For all docs  -->  docID: count_dict
corpora = defaultdict(dict)

# For all word --->  word: collection frequency
dictionary = defaultdict(int)

# Remove punctuations
trans_table = {ord(c): None for c in string.punctuation}    
stemmer = nltk.PorterStemmer()

# Tokenizer and Stemmer function
def tokenize(text):

    tokens = [word for word in nltk.word_tokenize(text.translate(trans_table)) if word not in set(stopwords.words('english'))] 
    stems = [stemmer.stem(item) for item in tokens]

    return stems

# open the file and extract data
with open('metadata.csv', encoding = "utf8", errors ='replace') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
    
        # access metadata
        cord_uid = row['cord_uid']
        title = row['title']
        abstract = row['abstract']

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


with open("dictionary.json","w",encoding="utf-8") as f:
        json.dump(dictionary,f, indent= 2, ensure_ascii = False)


# arrays for 
TF_IDF_vectors = []

for docID, counts in corpora.items():

    # Create count vector from count_dicts
    count_vector = []

    # Iterate over dictionary for each word
    for word, freq in dictionary.items():

        # Fetch tf and idf
        term_frequency = counts[word]
        inverted_doc_freq = math.log10(len(dictionary.keys())/freq)

        # If the term doesn't exist
        # then tf = 0
        count_vector.append(term_frequency * inverted_doc_freq)

    # Append vectors for the 2D array
    TF_IDF_vectors.append(count_vector)



tf_idf_matrix = pd.DataFrame(TF_IDF_vectors, index=list(corpora.keys()), columns=dictionary.keys())
print(tf_idf_matrix)

# pickled tf_idf matrix just in case
tf_idf_pickle = tf_idf_matrix.to_pickle("./tf_idf.pickle")


# A dictionary for query vectors 
# topicID: query_vector  
query_vectors = defaultdict(list)

for topic in odd_topics:

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
        inverted_doc_freq = math.log10(len(dictionary.keys())/freq)
    
        query_vector.append(term_frequency * inverted_doc_freq)

    
    query_vectors[topic["topic_id"]] = query_vector


# For each odd numbered topic 
for topic, q_vec in query_vectors.items():

    # Scores dict to be sorted 
    # docID: score
    scores = dict.fromkeys(list(corpora.keys()), 0)

    # Normalizing QUERY to a binary vector
    total = 0
    for word in q_vec:
        total += word**2

    qvec_norm = math.sqrt(total)

    for docID in list(corpora.keys()):
        
        doc_vec = tf_idf_matrix.loc[docID]

        # Normalizing DOCUMENT to a binary vector
        total = 0
        for word in doc_vec:
            total += word**2

        dvec_norm = math.sqrt(total)
        
        # Multiply vectors
        sums = 0
        for q, d in zip(q_vec, doc_vec):
            sums += q * d

        score = sums / (qvec_norm * dvec_norm)

        scores[docID] = score

    # Sort dictionary by values
    sorted_by_scores = dict(sorted(scores.items(), key=lambda item: item[1],reverse=True))