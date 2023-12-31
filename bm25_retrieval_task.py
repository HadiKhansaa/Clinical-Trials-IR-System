import os
import xml.etree.ElementTree as ET
import re
from bm25_inclusion_exclusion import tokenize_and_stem
from collections import defaultdict
import json
from compute_ndcg2 import computeNDCG
from search_trials_bert import is_eligible_for_trial
from precision_mrr import compute_avg_pr_mrr

TOPIC_ID = 1

def extract_queries(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    queries = []
    for topic in root.findall('topic'):
        text = re.sub(r'\s+', ' ', topic.text).strip()
        queries.append(text)

    return queries

def retrieve_and_score(query_terms, inverted_index):
    posting_lists = {term: inverted_index.get(term, {}) for term in query_terms}
    doc_scores = defaultdict(float)
    for term, postings in posting_lists.items():
        for doc, bm25_score in postings.items():
            doc_scores[doc] += bm25_score  # Add the bm25 score for the term

    return doc_scores

def combine_scores(i_scores, e_scores, d_scores, i_weight, e_weight, d_weight):
    combined_scores = defaultdict(float)

    for doc, score in i_scores.items():
        combined_scores[doc] += score * i_weight

    for doc, score in e_scores.items():
        combined_scores[doc] -= score * e_weight

    for doc, score in d_scores.items():
        combined_scores[doc] += score * d_weight

    return combined_scores

#fuction to find trial and load it
def find_trial(trial_id):
    global TOPIC_ID
    PATH_TO_TRIALS = f'trials_query{TOPIC_ID}'
    for file in os.listdir(os.path.join(PATH_TO_TRIALS)):
        if file[:-4] == trial_id:
            with open(os.path.join(PATH_TO_TRIALS, file), 'r') as f:
                try:
                    content = f.read()
                except:
                    continue
                return content
            

if __name__ == "__main__":
    # Extract queries
    extracted_queries = extract_queries("topics.xml")

    relevance_matrix = []
    for TOPIC_ID,query in enumerate(extracted_queries,1):
        # with open(f"files_query{TOPIC_ID}\\IC_bm25_index_SciSpacy_eis.json", 'r') as file:
        #     i_index = json.load(file)
        # with open(f"files_query{TOPIC_ID}\\EC_bm25_index_SciSpacy_eis.json", 'r') as file:
        #     e_index = json.load(file)
        # with open(f"files_query{TOPIC_ID}\\TSM_bm25_index_SciSpacy_eis.json", 'r') as file:
        #     d_index = json.load(file)
        with open(f"files_query{TOPIC_ID}\\IC_bm25_index.json", 'r') as file:
            i_index = json.load(file)
        with open(f"files_query{TOPIC_ID}\\EC_bm25_index.json", 'r') as file:
            e_index = json.load(file)
        with open(f"files_query{TOPIC_ID}\\TSM_bm25_index.json", 'r') as file:
            d_index = json.load(file)
        
        print(f"-------------------------------------------------------------------\nquery: {TOPIC_ID}\n")
        query_terms = tokenize_and_stem(query)
        i_scores = retrieve_and_score(query_terms, i_index)
        e_scores = retrieve_and_score(query_terms, e_index)
        d_scores = retrieve_and_score(query_terms, d_index)

        # Combine scores with weights
        total_scores = combine_scores(i_scores, e_scores, d_scores, 0.33, 0.33, 0.33)
        sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)

        '''to filter trials based on age and gender'''
        final_scores = defaultdict(float)
        for trial_xml,trial_score in total_scores.items():
            # Check if the trial is eligible for the topic
            if not is_eligible_for_trial(query, find_trial(trial_xml)):
                trial_score-=0.05 #penalty
            final_scores[trial_xml] = trial_score

        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        

        # Print combined scores for each document
        # for doc, score in sorted_scores[:10]:
        #     print(f"Document: {doc}, Score: {score}")


        json_file_path = 'relevance_feedback.json'

        # Load the relevance scores from the JSON file
        with open(json_file_path, 'r') as file:
            relevance_scores = json.load(file)

        document_ids = [doc for doc, _ in sorted_scores[:10]]

        # Create an array of relevance scores for the retrieved documents
        relevance_array = [relevance_scores.get(f"{TOPIC_ID}_{doc_id}", 0) for doc_id in document_ids]

        # compute ndcg
        print(relevance_array)
        relevance_matrix.append(relevance_array)
    
    ndcg_scores = computeNDCG(relevance_matrix)
    avg = 0
    for i,score in ndcg_scores.items():
        avg += score
    print("AVG@10 = ",avg/10)

    mrr , avg_prec = compute_avg_pr_mrr(relevance_matrix)
    print("AVG Precision@10 = ", avg_prec)
    print("AVG mrr = ", mrr)