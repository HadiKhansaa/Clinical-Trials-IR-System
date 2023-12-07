import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os

import xml.etree.ElementTree as ET

from compute_ndcg2 import computeNDCG

def is_eligible_for_trial(topic, trial_xml):
    gender, min_age, max_age = extract_trial_criteria(trial_xml)

    # Converting age ranges to integers, if applicable
    if min_age != 'N/A':
        min_age = int(min_age.split()[0])
    if max_age != 'N/A':
        max_age = int(max_age.split()[0])

    # Extracting age and gender from the topic
    age_match = re.search(r'\b(\d+)-year-old\b', topic, re.IGNORECASE)
    if age_match:
        topic_age = int(age_match.group(1))
    else:
        topic_age = -1
    topic_gender = 'male' if re.search(r'\bmale\b|\bman\b|\bboy\b', topic, re.IGNORECASE) else 'female'

    # Checking gender eligibility
    if gender.lower() != topic_gender and gender.lower() != 'all':
        return False

    # Checking age eligibility
    if min_age != 'N/A' and topic_age < min_age:
        return False
    if max_age != 'N/A' and topic_age > max_age:
        return False

    return True

def extract_trial_criteria(trial_xml):
    # Parse the XML content
    root = ET.fromstring(trial_xml)

    # Initialize default values
    gender = "N/A"
    min_age = "N/A"
    max_age = "N/A"

    # Extract gender
    gender_element = root.find('.//gender')
    if gender_element is not None:
        gender = gender_element.text

    # Extract minimum and maximum age
    min_age_element = root.find('.//minimum_age')
    if min_age_element is not None:
        min_age = min_age_element.text

    max_age_element = root.find('.//maximum_age')
    if max_age_element is not None:
        max_age = max_age_element.text

    return gender, min_age, max_age

#fuction to find trial and load it
def find_trial(trial_id):
    PATH_TO_TRIALS = 'trials_query3'
    # for folder in os.listdir(PATH_TO_TRIALS):
        # for sub_folder in os.listdir(os.path.join(PATH_TO_TRIALS, folder)):
    for file in os.listdir(os.path.join(PATH_TO_TRIALS)):
        if file[:-4] == trial_id:
            with open(os.path.join(PATH_TO_TRIALS, file), 'r') as f:
                try:
                    content = f.read()
                except:
                    continue
                return content

def load_topic(topic_id, xml_file_path):
    # Read the XML file
    with open(xml_file_path, 'r') as file:
        xml_content = file.read()
    
    # Parse the XML content
    root = ET.fromstring(xml_content)

    # Find the topic with the specified number
    topic_xpath = f".//topic[@number='{topic_id}']"
    topic_element = root.find(topic_xpath)

    if topic_element is not None:
        return topic_element.text
    else:
        return "Topic not found."

# Function to load embeddings from a JSON file
def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        embeddings = json.load(file)
    return embeddings

# Function to compute the top-10 most similar trials for each topic
def compute_top_10_similarities(topic_embeddings, trial_embeddings):
    topic_ids = list(topic_embeddings.keys())
    trial_ids = list(trial_embeddings.keys())
    results = {}

    for topic_id in topic_ids:
        similarities = []
        topic_emb = np.array(topic_embeddings[topic_id]).reshape(1, -1)
        topic = load_topic(int(topic_id.split('_')[1]), "topics.xml")
        for trial_id in trial_ids:
            trial_xml = find_trial(trial_id)

            trial_emb = np.array(trial_embeddings[trial_id]).reshape(1, -1)
            sim_score = cosine_similarity(topic_emb, trial_emb)[0][0]

            # Check if the trial is eligible for the topic
            if not is_eligible_for_trial(topic, trial_xml):
                sim_score-=0.005 #penalty

            similarities.append((trial_id, sim_score))

        # Sort by similarity score and get top 10
        top_10_trials = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        results[topic_id] = top_10_trials

    return results

def rerank(top_10_results, trial_embeddings_summary_volun, topic_embeddings_summary_volun):
    topic_id = "topic_1"
    trial_ids = [trial_id for trial_id, score in top_10_results]
    results = {}

    similarities = []
    topic_emb = np.array(topic_embeddings_summary_volun[topic_id]).reshape(1, -1)
    topic = load_topic(int(topic_id.split('_')[1]), "topics.xml")
    for trial_id in trial_ids:
        trial_xml = find_trial(trial_id)

        trial_emb = np.array(trial_embeddings_summary_volun[trial_id]).reshape(1, -1)
        sim_score = cosine_similarity(topic_emb, trial_emb)[0][0]

        # Check if the trial is eligible for the topic
        if not is_eligible_for_trial(topic, trial_xml):
            sim_score-=0.005 #penalty

        similarities.append((trial_id, sim_score))

        # Sort by similarity score and get top 10
        top_10_trials = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        results[topic_id] = top_10_trials

    return results


json_file_path = 'relevance_feedback.json'

# Load the relevance scores from the JSON file
with open(json_file_path, 'r') as file:
    relevance_scores = json.load(file)

topic_embeddingsClinical = load_embeddings('files_query3\\topic_embeddings_Clinical.json')
trial_embeddingsClinical = load_embeddings('files_query3\\trial_embeddings_Clinical_q3.json')

top_10_resultsClinical = compute_top_10_similarities(topic_embeddingsClinical, trial_embeddingsClinical)

#before reranking
# Print the results
print("Before reranking:")
for topic, trials in top_10_resultsClinical.items():
    if topic == "topic_2":
        break
    print(f"Topic {topic}:")
    for trial, score in trials:
        print(f"  - Trial ID: {trial}, Similarity: {score:.4f}")
    print()

    document_ids = [doc for doc, _ in trials]

    # Create an array of relevance scores for the retrieved documents
    relevance_array = [relevance_scores.get(f"3_{doc_id}", 0) for doc_id in document_ids]

    # compute ndcg
    print(relevance_array)
    computeNDCG([relevance_array])

topic_embeddings = load_embeddings('files_query3\\topic_embeddings_BERT.json')
trial_embeddings = load_embeddings('files_query3\\trial_embeddings_BERT_q3.json')

top_10_results = rerank(top_10_resultsClinical["topic_1"], trial_embeddings, topic_embeddings)

# Print the results
print("After reranking:")
for topic, trials in top_10_results.items():
    if topic == "topic_2":
        break
    print(f"Topic {topic}:")
    for trial, score in trials:
        print(f"  - Trial ID: {trial}, Similarity: {score:.4f}")
    print()

    document_ids = [doc for doc, _ in trials]

    # Create an array of relevance scores for the retrieved documents
    relevance_array = [relevance_scores.get(f"3_{doc_id}", 0) for doc_id in document_ids]

    # compute ndcg
    print(relevance_array)
    computeNDCG([relevance_array])