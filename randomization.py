import os
import random
import re
import subprocess as sp
from preprocessing import *

def getResultsData(file_name):
    '''
    -> Reads a results file and maps rankings of
    a topic with its topic ID.
    '''
    result_data = defaultdict(dict)
    with open(f"tra/{file_name}_results.txt", "r") as f:
        lines = f.readlines()
        doc_score_dict = defaultdict(float)
        last_topic = "1"
        for line in lines:
            tokens = line.encode().decode("utf-8").split()
            topic = tokens[0]
            docID = tokens[2]
            score = tokens[4]
            # Appends dictionary if all the topic related
            # documents are added
            if topic != last_topic:
                result_data[last_topic] = doc_score_dict
                last_topic = topic
                doc_score_dict = defaultdict(float)
            doc_score_dict[docID] = score

    return result_data

def calculatePerformanceValues(file_name):
    '''
    -> Runs the provided trec_eval evaluation tool
    via subprocess and retrieves the results within a
    cleansed form.
    '''
    result = sp.run([f"./trec_eval -m map -m ndcg -m P.5,10 relevancy-judgements.txt tra/{file_name}_results.txt"], stdout=sp.PIPE, shell=True)
    result = result.stdout.decode("utf-8", errors="replace")    

    tokens = result.split()
    tokens = [float(token) for ix, token in enumerate(tokens) if ix % 3 == 2]
    return tuple(tokens)

def randomizationTest(first_sys_name, second_sys_name):
    '''
    -> Applies randomization test to evaluate the statistical
    side of our models.
    '''
    all_first_results = getResultsData(first_sys_name)
    all_second_results = getResultsData(second_sys_name)
    
    # initial_differences
    initial_first_performance_values = calculatePerformanceValues(first_sys_name)
    inital_second_performance_values = calculatePerformanceValues(second_sys_name)
    
    initial_difference_MAP = abs(initial_first_performance_values[0] - inital_second_performance_values[0])
    initial_difference_P5 = abs(initial_first_performance_values[1] - inital_second_performance_values[1])
    initial_difference_P10 = abs(initial_first_performance_values[2] - inital_second_performance_values[2])
    initial_difference_NDCG = abs(initial_first_performance_values[3] - inital_second_performance_values[3])
    
    counter_MAP = 0
    counter_P5 = 0
    counter_P10 = 0
    counter_NDCG = 0

    # We've decided to run the algorithm 1000 times.
    R = 1000
    # We've selected our seed to provide consecutive
    # repeatable outputs.
    random.seed(47)

    for ix in range(0, R):
        if ix % 100 == 0:
            print(f"Iteration: {ix}")

        for topic in all_first_results.keys():
            first_result = all_first_results[topic]
            second_result = all_second_results[topic]

            # We've swapped the rankings of the same topic
            # between two models
            # To store current state, we've created two
            # temporary files to preserve data and calculate
            # absolute difference between those two outputs
            if random.random() <= 0.5:
                write_to_file(topic, first_result, f"temp_{first_sys_name}")
                write_to_file(topic, second_result, f"temp_{second_sys_name}")
            else:
                write_to_file(topic, second_result, f"temp_{first_sys_name}")
                write_to_file(topic, first_result, f"temp_{second_sys_name}")

        # subprocess get new value

        current_first_performance_values = calculatePerformanceValues(f"temp_{first_sys_name}")
        current_second_performance_values = calculatePerformanceValues(f"temp_{second_sys_name}")

        # Calculating the current values
        current_difference_MAP = abs(current_first_performance_values[0] - current_second_performance_values[0])
        current_difference_P5 = abs(current_first_performance_values[1] - current_second_performance_values[1])
        current_difference_P10 = abs(current_first_performance_values[2] - current_second_performance_values[2])
        current_difference_NDCG = abs(current_first_performance_values[3] - current_second_performance_values[3])
        
        if current_difference_MAP >= initial_difference_MAP:
            counter_MAP += 1 
        if current_difference_P5 >= initial_difference_P5:
            counter_P5 += 1 
        if current_difference_P10 >= initial_difference_P10:
            counter_P10 += 1 
        if current_difference_NDCG >= initial_difference_NDCG:
            counter_NDCG += 1 

        # Removing the temporary files
        sp.run([f"rm -rf tra/temp_{first_sys_name}_results.txt"], shell=True)
        sp.run([f"rm -rf tra/temp_{second_sys_name}_results.txt"], shell=True)

    # Calculating the final p values for each measure
    p_value_P5 = (counter_P5 + 1) / (R + 1)
    p_value_P10 = (counter_P10 + 1) / (R + 1)
    p_value_NDCG = (counter_NDCG + 1) / (R + 1)
    p_value_MAP = (counter_MAP + 1) / (R + 1)

    print(first_sys_name, "---", second_sys_name)
    print()
    print("p value for MAP: \t", p_value_MAP)
    print("p value for P5: \t", p_value_P5)
    print("p value for P10: \t", p_value_P10)
    print("p value for NDCG: \t", p_value_NDCG)

if __name__ == "__main__":
    randomizationTest(sys.argv[1], sys.argv[2])
