import nltk
import os
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Ensure you have the necessary nltk data
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()

nlp = spacy.load("en_ner_bc5cdr_md")

def filter_medical_terms(text):
    doc = nlp(text)
    # print(list(doc.ents))
    # Filter out entities that are identified as medical terms
    medical_terms = [ent.text for ent in doc.ents]
    return medical_terms

def extract_components(content):
    # Extracting the required components using regular expressions
    inc_match = re.search(r'<textblock>.*?Inclusion Criteria:(.*?)Exclusion Criteria:(.*?)</textblock>', content, re.DOTALL)
    title_match = re.search(r'<brief_title>(.*?)</brief_title>', content)
    summary_match = re.search(r'<brief_summary>.*?<textblock>(.*?)</textblock>.*?</brief_summary>', content, re.DOTALL)
    desc_match = re.search(r'<detailed_description>.*?<textblock>(.*?)</textblock>.*?</detailed_description>', content, re.DOTALL)
    keyword_matches = re.findall(r'<keyword>(.*?)</keyword>', content)
    mesh_matches = re.findall(r'<mesh_term>(.*?)</mesh_term>', content)
    # Extracting text from matches or setting defaults
    title = title_match.group(1) if title_match else ''
    summary = summary_match.group(1) if summary_match else ''
    desc = desc_match.group(1) if desc_match else ''
    inc_criteria = inc_match.groups()[0] if inc_match else ''
    exc_criteria = inc_match.groups()[1] if inc_match else ''
    keywords = ' '.join(keyword_matches)
    mesh_terms = ' '.join(mesh_matches)

    # Concatenating all extracted components
    return f"{title} {summary} {desc} {inc_criteria} {exc_criteria} {keywords} {mesh_terms}"

def create_inverted_index(feature_names, tfidf_matrix, docs):
    index = {}
    dense_matrix = tfidf_matrix.todense()
    for i, term in enumerate(feature_names):
        term_scores = {}
        for doc_id, score in zip(docs, dense_matrix[:, i].A1):
            if score > 0:
                term_scores[doc_id] = score
        # Sort the documents by TF-IDF score in descending order
        sorted_term_scores = dict(sorted(term_scores.items(), key=lambda item: item[1], reverse=True))
        index[term] = sorted_term_scores
    return index

if __name__ == "__main__":
    # Initialize three separate vectorizers
    vectorizer_index = TfidfVectorizer()

    trials_content = []
    doc_ids = []

    for topic_id in range(1,11):

        PATH_TO_TRIALS = f'trials_query{topic_id}'
        print("proccesing topic: ",topic_id)

        i = 1
        # Iterate over folders and files
        for file in os.listdir(os.path.join(PATH_TO_TRIALS)):
            with open(os.path.join(PATH_TO_TRIALS, file), 'r') as f:
                if file.endswith('.xml'):
                    try:
                        content = f.read()
                    except:
                        continue
                    # content = extract_components(content)
                    if content !='':
                        trials_content.append(content)
                    doc_ids.append(file[:-4])  # Assuming file names are the document IDs
                    print(f"Processed {i} files")  # Print progress
                    i += 1

        # Fit and transform the texts
        tfidf_content = vectorizer_index.fit_transform(trials_content)

        # Create inverted indexes
        inverted_index = create_inverted_index(vectorizer_index.get_feature_names_out(), tfidf_content, doc_ids)

        # Save to JSON files
        with open(f'files_query{topic_id}\\baseline_tfidf_index.json', 'w') as f:
            json.dump(inverted_index, f)
