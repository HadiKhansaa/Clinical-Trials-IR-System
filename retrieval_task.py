import xml.etree.ElementTree as ET
import re
from create_tfidf_inclusion_exclusion import tokenize_and_stem
from collections import defaultdict
import json

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

def retrieve_and_score(query, inverted_index):
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

    with open("inclusion_criteria_index.json", 'r') as file:
        i_index = json.load(file)
    with open("exclusion_criteria_index.json", 'r') as file:
        e_index = json.load(file)
    with open("title_desc_mesh_index.json", 'r') as file:
        d_index = json.load(file)
    
    for query in extracted_queries:
        print(f"-------------------------------------------------------------------\nquery: {query}\n")
        # Tokenize and preprocess the query terms
        query_terms = tokenize_and_stem(query)
        #search against i_index -> score i_score, weight = 33
        i_scores = retrieve_and_score(query_terms, i_index)
        #search against e_index -> score e_score, weight = 44
        e_scores = retrieve_and_score(query_terms, e_index)
        #search against d_index -> score d_score, weight = 23
        d_scores = retrieve_and_score(query_terms, d_index)

        # Combine scores with weights
        total_scores = combine_scores(i_scores, e_scores, d_scores, 0.33, 0.44, 0.23)
        
        # Sort the scores in decreasing order
        sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)

        # Print combined scores for each document
        for doc, score in sorted_scores:
            print(f"Document: {doc}, Score: {score}")

