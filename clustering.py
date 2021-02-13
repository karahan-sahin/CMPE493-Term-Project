import random as rn
from collections import defaultdict

rn_seeds = 47
rn.seed(rn_seeds)

epoch = 0
seeds = defaultdict(dict)
cluster_doc_map = defaultdict(list)
min_seed_length = 9

# {"1": [docID, sda, da  ] }
# metric_dictionary[cluster_doc_map[1][i]]

EPOCH_LIMIT = 3

def select_initial_seeds(metric_dictionary):
    global seeds, min_seed_length
    non_labeled_documents = list(metric_dictionary.values())
    for i in range(25):
        seed_doc = rn.choice(non_labeled_documents)
        while seed_doc in seeds.values():
            seed_doc = rn.choice(non_labeled_documents)

        # if len(seed_doc.values()) < min_seed_length:
        #     min_seed_length = len(seed_doc.values())
        # seed_doc = dict(sorted(seed_doc.items(), key=lambda item: item[1],reverse=True))

        # top_nine_words_centroid = list(seed_doc.keys())[:min_seed_length]
        # seed_doc = {key: seed_doc[key] for key in top_nine_words_centroid}
        seeds[i] = seed_doc

def calculate_new_centroids(metric_dictionary):
    print("calculate_new_centroids")
    global cluster_doc_map, seeds
    temp_seeds = defaultdict(dict)
    for i in range(25):
        current_cluster_docsID = cluster_doc_map[i]
        new_centroid = vector_average(current_cluster_docsID, metric_dictionary)
        temp_seeds[i] = new_centroid
    del seeds
    seeds = temp_seeds
    print("calculate_new_centroids done")

def vector_average(docList, metric_dictionary):
    global min_seed_length
    new_centroid = defaultdict(float)
    for docID in docList:
        for word, weight in metric_dictionary[docID].items():
            new_centroid[word] += weight

    # DOC1 "HELLO WORLD" {HELLO : 0.36, WORLD: 0.46}
    # DOC2 "HELLO DUDE"  {HELLO : 0.24, DUDE: 0.18 }
    # CENTROID           {HELLO: 0.3, WORLD: 0.23, DUDE: 0.09}
    for word, weight in new_centroid.items():
        new_centroid[word] = weight / len(docList)

    new_centroid = dict(sorted(new_centroid.items(), key=lambda item: item[1],reverse=True))
    top_twenty_words_centroid = list(new_centroid.keys())[:20]
    new_centroid = {key: new_centroid[key] for key in top_twenty_words_centroid}
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
        sums += first_doc[intersect_key] * second_doc[intersect_key]

    score = sums / (first_norm * second_norm)
    return score
    
def is_seeds_same(previous_seeds):
    global seeds
    print("Seed check")
    for i in range(25):
        prev_seed_hash = hash(frozenset(previous_seeds[i]))
        curr_seed_hash = hash(frozenset(seeds[i]))
        if prev_seed_hash != curr_seed_hash:
            return False
    print("Seed check done")
    return True

def finalize_clustering(metric_dictionary):
    global seeds, epoch
    count = 0
    while True:
        previous_seeds = seeds
        print("Iteration start.")
        iterate_clustering(metric_dictionary)
        print("Iteration end.")
        count += 1
        print(f"Iteration: {count}")
        if is_seeds_same(previous_seeds):
            epoch += 1
            if epoch > EPOCH_LIMIT:
                break
        else:
            epoch = 0

def iterate_clustering(metric_dictionary):
    global cluster_doc_map
    del cluster_doc_map
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

    calculate_new_centroids(metric_dictionary)

def compose_clusters(metric_dictionary):
    select_initial_seeds(metric_dictionary) 
    finalize_clustering(metric_dictionary)

    return seeds, cluster_doc_map
