import nltk
import os
import re
import xml.etree.ElementTree as ET
import json
import torch
from transformers import BertTokenizer, BertModel
from nltk.stem.porter import PorterStemmer

# Ensure you have the necessary nltk data
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()

import xml.etree.ElementTree as ET
import re

import xml.etree.ElementTree as ET
import re

def extract_components(xml_content):
    # Parse the XML content
    root = ET.fromstring(xml_content)

    # Extract title
    title = root.findtext('.//brief_title', default='').strip()

    # Extract detailed description
    detailed_desc = root.findtext('.//brief_summary/textblock', default='').strip()
    detailed_desc = re.sub(r'\s+', ' ', detailed_desc)  # Remove excessive whitespace

    # Extract inclusion and exclusion criteria
    criteria_text = root.findtext('.//eligibility/criteria/textblock', default='')
    # inc_criteria, exc_criteria = '', ''
    # if criteria_text:
    #     inc_match = re.search(r'inclusion criteria:(.*?)(exclusion criteria:|$)', criteria_text, re.DOTALL | re.IGNORECASE)
    #     exc_match = re.search(r'exclusion criteria:(.*)', criteria_text, re.DOTALL | re.IGNORECASE)

    #     inc_criteria = inc_match.group(1).strip() if inc_match else ''
    #     exc_criteria = exc_match.group(1).strip() if exc_match else ''

    #     # Clean up criteria formatting
    #     inc_criteria = re.sub(r'\s+', ' ', inc_criteria)
    #     exc_criteria = re.sub(r'\s+', ' ', exc_criteria)
    criteria_text = re.sub(r'\s+', ' ', criteria_text)
    # Extract mesh terms
    mesh_terms = [mesh_term.text for mesh_term in root.findall('.//condition_browse/mesh_term')]
    mesh_terms = ', '.join(mesh_terms)

    # Extract gender
    gender = root.findtext('.//gender', default='').strip()

    # Extract minimum and maximum age
    minimum_age = root.findtext('.//minimum_age', default='').strip()
    maximum_age = root.findtext('.//maximum_age', default='').strip()

    return title, detailed_desc, criteria_text, mesh_terms, gender, minimum_age, maximum_age




# Function to create embeddings using BERT
def create_bert_embeddings(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)

    embeddings = output.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

if __name__ == "__main__":
    # PATH_TO_TRIALS = 'C:\\Users\\Hp\\Documents\\CMPS M\\CMPS 365\\project\\trecs\\trials'
    PATH_TO_TRIALS = 'topic1_trials'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    trial_embeddings = {}

    # get documents with relevence feedback
    with open('documents.json', 'r') as file:
        docs = json.load(file)
    documents = set(docs)

    # print(documents)

    i = 1
    for folder in os.listdir(PATH_TO_TRIALS):
        for file in os.listdir(os.path.join(PATH_TO_TRIALS, folder)):
            with open(os.path.join(PATH_TO_TRIALS, folder, file), 'r') as f:
                if file.endswith('.xml') and file[:-4] in documents: # check if file is in documents with relevence feedback
                    try:
                        content = f.read()
                    except:
                        continue

                    title, detailed_desc, criteria_text, mesh_terms, gender, minimum_age, maximum_age = extract_components(content)
                    if detailed_desc == '':
                        detailed_desc = "No brief summary available"
                    if criteria_text == '':
                        criteria_text = "No criteria available"
                    if mesh_terms == '':
                        mesh_terms = "No mesh terms available"
    
                    combined_text = f"Title: {title}\n\nSummary: {detailed_desc}\n\nCriteria: {criteria_text}\n\nMesh Terms: {mesh_terms}\n\nGender Required: {gender}\n\nMinimum Age: {minimum_age}\n\nMaximum Age: {maximum_age}\n\n"

                    # print(file[:-4])
                    # print(combined_text)

                    doc_id = file[:-4]  # Assuming file names are the document IDs
                    trial_embeddings[doc_id] = create_bert_embeddings(combined_text, tokenizer, model)

                    print(f"Processed {i} files")  # Print progress
                    i += 1

    # Save embeddings to a JSON file
    with open('trial_embeddings.json', 'w') as f:
        json.dump(trial_embeddings, f)
