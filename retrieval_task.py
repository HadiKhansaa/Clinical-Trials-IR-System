import os
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from compute_ndcg2 import computeNDCG
from search_trials_bert import is_eligible_for_trial

# Ensure you have the necessary nltk data
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()

#fuction to find trial and load it
def find_trial(trial_id):
    PATH_TO_TRIALS = 'trials_query10'
    for file in os.listdir(os.path.join(PATH_TO_TRIALS)):
        if file[:-4] == trial_id:
            with open(os.path.join(PATH_TO_TRIALS, file), 'r') as f:
                try:
                    content = f.read()
                except:
                    continue
                return content

def extract_queries(file_path):
    # Parse the XML content from the file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize a list to store the extracted queries
    queries = []

    # Iterate through each 'topic' element in the XML
    for topic in root.findall('topic'):
        # Extract the text content of the topic
        # Use regex to remove unnecessary whitespace
        text = re.sub(r'\s+', ' ', topic.text).strip()
        # Append the cleaned text to the queries list
        queries.append(text)

    return queries

def retrieve_and_score(query_terms, inverted_index): #changed
    # Retrieve the posting lists for each query term
    posting_lists = {term: inverted_index.get(term, {}) for term in query_terms}
    
    # Compute the score for each document
    doc_scores = defaultdict(float)
    for term, postings in posting_lists.items():
        for doc, tf_idf_score in postings.items():
            doc_scores[doc] += tf_idf_score  # Add the tf-idf score for the term

    return doc_scores

def combine_scores(i_scores, e_scores, d_scores, i_weight, e_weight, d_weight):
    combined_scores = defaultdict(float)

    # Combine scores from i_index
    for doc, score in i_scores.items():
        combined_scores[doc] += score * i_weight

    # Combine scores from e_index, with a negative weight
    for doc, score in e_scores.items():
        combined_scores[doc] -= score * e_weight

    # Combine scores from d_index
    for doc, score in d_scores.items():
        combined_scores[doc] += score * d_weight

    return combined_scores

if __name__ == "__main__":
    # Extract queries
    extracted_queries = extract_queries("topics.xml")

    with open("files_query10\\IC_tfidf_index.json", 'r') as file:
        i_index = json.load(file)
    with open("files_query10\\EC_tfidf_index.json", 'r') as file:
        e_index = json.load(file)
    with open("files_query10\\TSM_tfidf_index.json", 'r') as file:
        d_index = json.load(file)
    
    for i,query in enumerate(extracted_queries):
        if(i>0):
            break
        print(f"-------------------------------------------------------------------\nquery: {query}\n")
        # Tokenize and preprocess the query terms
        query_terms = query.lower().split()
        #search against i_index -> score i_score, weight = 33
        i_scores = retrieve_and_score(query_terms, i_index)
        #search against e_index -> score e_score, weight = 44
        e_scores = retrieve_and_score(query_terms, e_index)
        #search against d_index -> score d_score, weight = 23
        d_scores = retrieve_and_score(query_terms, d_index)

        # Combine scores with weights
        total_scores = combine_scores(i_scores, e_scores, d_scores, 0.33, 0.44, 0.23)
        sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
        
        '''to filter trials based on age and gender'''
        final_scores = defaultdict(float)
        for trial_xml,trial_score in total_scores.items():
            # Check if the trial is eligible for the topic
            if not is_eligible_for_trial(query, find_trial(trial_xml)):
                # print(f"Trial {trial_id} is not eligible for topic {topic_id}")
                trial_score-=0.05 #penalty
                # trial_score-=0.9 #penalty
            final_scores[trial_xml] = trial_score

        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Print combined scores for each document
        for doc, score in sorted_scores[:10]:
            print(f"Document: {doc}, Score: {score}")



        json_file_path = 'relevance_feedback.json'

        # Load the relevance scores from the JSON file
        with open(json_file_path, 'r') as file:
            relevance_scores = json.load(file)

        document_ids = [doc for doc, _ in sorted_scores[:10]]

        # Create an array of relevance scores for the retrieved documents
        relevance_array = [relevance_scores.get(f"10_{doc_id}", 0) for doc_id in document_ids]

        # compute ndcg
        print(relevance_array)
        computeNDCG([relevance_array])