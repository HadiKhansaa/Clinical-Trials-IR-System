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

if __name__ == "__main__":
    # Extract queries
    extracted_queries = extract_queries("topics.xml")

    with open("files_query10\\baseline_tfidf_index.json", 'r') as file:
        i_index = json.load(file)
    
    for i,query in enumerate(extracted_queries):
        if(i>0):
            break
        print(f"-------------------------------------------------------------------\nquery: {query}\n")
        # Tokenize and preprocess the query terms
        query_terms = query.lower().split()
        total_scores = retrieve_and_score(query_terms, i_index)

        sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Print combined scores for each document
        # for doc, score in sorted_scores[:10]:
        #     print(f"Document: {doc}, Score: {score}")



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