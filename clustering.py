import random as rn
from collections import defaultdict

epoch = 0
seeds = defaultdict(dict)
cluster_doc_map = defaultdict(list)

# {"1": [docID, sda, da  ] }
# metric_dictionary[cluster_doc_map[1][i]]

EPOCH_LIMIT = 3

def select_initial_seeds(metric_dictionary):
    global seeds
    non_labeled_documents = metric_dictionary.values()
    for i in range(25):
        seed_doc = rn.choice(non_labeled_documents)
        while seed_doc in seeds.values():
            seed_doc = rn.choice(non_labeled_documents)
        seeds[i] = seed_doc

def calculate_new_centroids(metric_dictionary):
    global cluster_doc_map, seeds
    temp_seeds = defaultdict(dict)
    for i in range(25):
        current_cluster_docsID = cluster_doc_map[i]
        new_centroid = vector_average(current_cluster_docsID, metric_dictionary)
        temp_seeds[i] = new_centroid
    seeds = temp_seeds

def vector_average(docList, metric_dictionary):
    new_centroid = defaultdict(float)
    for docID in docList:
        for word, weight in metric_dictionary[docID]:
            new_centroid[word] += weight

    # DOC1 "HELLO WORLD" {HELLO : 0.36, WORLD: 0.46}
    # DOC2 "HELLO DUDE"  {HELLO : 0.24, DUDE: 0.18 }
    # CENTROID           {HELLO: 0.3, WORLD: 0.23, DUDE: 0.09}
    for word, weight in new_centroid.items():
        new_centroid[word] = weight / len(docList)

    return new_centroid

def calculate_norm(doc):
    norm = 0.0
    for val in doc.values():
        norm += val ** 2
    return norm ** 0.5

def calc_cosine_similarity(first_doc, second_doc):
    first_norm = calculate_norm(first_doc)
    second_norm = calculate_norm(second_doc)

    sums = 0
    first_doc_keys = set(first_doc.keys())
    second_doc_keys = set(second_doc.keys())
    intersected_keys = first_doc_keys.intersection(second_doc_keys)

    for intersect_key in list(intersected_keys):
        sums += first_doc_keys[intersect_key] * second_doc_keys[intersect_key]

    score = sums / (first_norm * second_norm)
    return score
    
def is_seeds_same(previous_seeds):
    global seeds
    for i in range(25):
        if not len(previous_seeds[i]) == len(seeds[i]):
            return False
        for word in list(set(previous_seeds[i].keys()).union(set(seeds[i].keys()))):
            if previous_seeds[word] != seeds[word]:
                return False
            else:
                continue
    return True

def finalize_clustering(metric_dictionary):
    global seeds, epoch
    while True:
        previous_seeds = seeds
        iterate_clustering(metric_dictionary)

        if is_seeds_same(previous_seeds):
            epoch += 1
            if epoch > EPOCH_LIMIT:
                break
        else:
            epoch = 0

def iterate_clustering(metric_dictionary):
    global cluster_doc_map
    cluster_doc_map = defaultdict(list)
    for docId, docMetric in metric_dictionary.items():

        min_distance = 1.0
        min_centroid_id = 0

        for centroid_id, centroid_val in seeds.items():
            distance = 1 - calc_cosine_similarity(docMetric, centroid_val)
            if distance <= min_distance:
                min_distance = distance
                min_centroid_id = centroid_id

        cluster_doc_map[min_centroid_id].append(docId)

    calculate_new_centroids()

def compose_clusters(metric_dictionary):
    select_initial_seeds(metric_dictionary) 
    finalize_clustering(metric_dictionary)

    return seeds, cluster_doc_map
