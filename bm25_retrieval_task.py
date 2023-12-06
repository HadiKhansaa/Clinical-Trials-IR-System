import os
import xml.etree.ElementTree as ET
import re
from bm25_inclusion_exclusion import tokenize_and_stem
from collections import defaultdict
import json
from compute_ndcg import computeNDCG
from search_trials_bert import is_eligible_for_trial

#fuction to find trial and load it
def find_trial(trial_id):
    PATH_TO_TRIALS = 'topic1_trials'
    for folder in os.listdir(PATH_TO_TRIALS):
        for file in os.listdir(os.path.join(PATH_TO_TRIALS, folder)):
            if file[:-4] == trial_id:
                with open(os.path.join(PATH_TO_TRIALS, folder, file), 'r') as f:
                    try:
                        content = f.read()
                    except:
                        continue
                    return content

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

if __name__ == "__main__":
    # Extract queries
    extracted_queries = extract_queries("topics.xml")

    with open("inclusion_criteria_index_bm25.json", 'r') as file:
        i_index = json.load(file)
    with open("exclusion_criteria_index_bm25.json", 'r') as file:
        e_index = json.load(file)
    with open("title_desc_mesh_index_bm25.json", 'r') as file:
        d_index = json.load(file)
    
    for query in extracted_queries:
        print(f"-------------------------------------------------------------------\nquery: {query}\n")
        query_terms = tokenize_and_stem(query)
        i_scores = retrieve_and_score(query_terms, i_index)
        e_scores = retrieve_and_score(query_terms, e_index)
        d_scores = retrieve_and_score(query_terms, d_index)

        # Combine scores with weights
        total_scores = combine_scores(i_scores, e_scores, d_scores, 0.33, 0.44, 0.23)
        # sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
        
        '''to filter trials based on age and gender'''
        final_scores = defaultdict(float)
        for trial_xml,trial_score in total_scores.items():
            # Check if the trial is eligible for the topic
            if not is_eligible_for_trial(query, find_trial(trial_xml)):
                # print(f"Trial {trial_id} is not eligible for topic {topic_id}")
                continue
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
        relevance_array = [relevance_scores.get(f"1_{doc_id}", 0) for doc_id in document_ids]

        # compute ndcg
        print(relevance_array)
        computeNDCG([relevance_array])

'''
50 trials
bm25 without filtering out age and gender
[1, 1, 2, 2, 0, 2, 0, 1, 0, 0] 0.816
bm25+sciSPacy (exc) without filtering out age and gender 
[2, 1, 0, 0, 1, 1, 0, 0, 0, 0] 0.924
bm25+sciSPacy (inc+exc) without filtering out age and gender 
[1, 2, 0, 0, 0, 1, 0, 0, 0, 0] 0.93
bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 
[0, 0, 2, 0, 2, 0, 1, 2, 0, 0] 0.546
bm25 with filtering out age and gender
[1, 1, 2, 2, 0, 2, 0, 0, 0, 0] 0.813
tfidf without filtering
[0, 1, 1, 2, 2, 2, 0, 1, 1, 0] 0.708
tfidf after filtering
[0, 1, 2, 2, 2, 0, 1, 1, 0, 0] 0.73
BERT with/without filtering
[1, 0, 0, 2, 0, 1, 0, 0, 0, 0] 0.657
'''

'''
35 trials
bm25 without filtering out age and gender
[1, 1, 0, 0, 0, 2, 0, 2, 1, 0] 0.675
bm25+sciSPacy (exc) without filtering out age and gender 
[2, 1, 0, 1, 2, 0, 0, 1, 1, 0] 0.84
bm25+sciSPacy (inc+exc) without filtering out age and gender 
[2, 1, 0, 1, 0, 1, 0, 0, 0, 0] 0.94
bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 
[2, 1, 0, 1, 2, 0, 0, 0, 0, 0] 0.85
bm25 with filtering out age and gender
[1, 0, 0, 2, 0, 2, 1, 0, 0, 0] 0.61
tfidf without filtering
[0, 0, 2, 0, 0, 0, 0, 1, 2, 0] 0.48
tfidf after filtering
[0, 2, 0, 0, 0, 1, 2, 0, 1, 1] 0.668
BERT with/without filtering
[1, 0, 0, 2, 0, 1, 0, 0, 0, 1] 0.65
'''

'''
brief summary + 50 trials

bm25 without filtering out age and gender
[2, 1, 2, 0, 0, 0, 0, 0, 0, 0] 0.92
bm25+sciSPacy (exc) without filtering out age and gender 

bm25+sciSPacy (inc+exc) without filtering out age and gender 

bm25+sciSPacy (tdm+inc+exc) without filtering out age and gender 

bm25 with filtering out age and gender
[2, 1, 2, 0, 0, 0, 0, 1, 2, 0] 0.84
tfidf without filtering

tfidf after filtering

BERT
[0, 2, 1, 0, 0, 1, 2, 2, 0, 1] 0.71
'''