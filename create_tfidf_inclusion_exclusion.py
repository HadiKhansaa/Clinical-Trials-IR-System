import nltk
import os
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure you have the necessary nltk data
nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

def tokenize_and_stem(text):
    # Tokenize by word
    tokens = word_tokenize(text)
    # Filter out any tokens not containing letters
    filtered_tokens = [word for word in tokens if word.isalpha()]
    # Stem the filtered tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# Function to extract different components using regex
def extract_components(content):
    inc_match = re.search(r'<textblock>.*?Inclusion Criteria:(.*?)Exclusion Criteria:(.*?)</textblock>', content, re.DOTALL)
    title_match = re.search(r'<brief_title>(.*?)</brief_title>', content)
    desc_match = re.search(r'<detailed_description>.*?<textblock>(.*?)</textblock>.*?</detailed_description>', content, re.DOTALL)
    mesh_match = re.findall(r'<mesh_term>(.*?)</mesh_term>', content)

    inc_criteria = inc_match.groups()[0] if inc_match else ''
    exc_criteria = inc_match.groups()[1] if inc_match else ''
    if exc_criteria == '':
        inc_match = re.search(r'<textblock>.*?Inclusion Criteria:(.*?)</textblock>', content, re.DOTALL)
        inc_criteria = inc_match.groups()[0] if inc_match else ''
    title = title_match.group(1) if title_match else ''
    desc = desc_match.group(1) if desc_match else ''
    mesh_terms = ' '.join(mesh_match)

    return inc_criteria, exc_criteria, f"{title} {desc} {mesh_terms}"

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
    vectorizer_inclusion = TfidfVectorizer(tokenizer=tokenize_and_stem)
    vectorizer_exclusion = TfidfVectorizer(tokenizer=tokenize_and_stem)
    vectorizer_title_desc_mesh = TfidfVectorizer(tokenizer=tokenize_and_stem)

    inclusion_criteria = []
    exclusion_criteria = []
    title_desc_mesh = []
    doc_ids = []

    PATH_TO_TRIALS = 'testing_trials'

    i = 1
    # Iterate over folders and files
    for folder in os.listdir(PATH_TO_TRIALS):
        for file in os.listdir(os.path.join(PATH_TO_TRIALS, folder)):
            with open(os.path.join(PATH_TO_TRIALS, folder, file), 'r') as f:
                if file.endswith('.xml'):
                    content = f.read()
                    inc, exc, tdm = extract_components(content)
                    if inc !='':
                        inclusion_criteria.append(inc)
                    if exc != '':
                        exclusion_criteria.append(exc)
                    if tdm != '':
                        title_desc_mesh.append(tdm)
                    doc_ids.append(file[:-4])  # Assuming file names are the document IDs
                    print(f"Processed {i} files")  # Print progress
                    i += 1

    # Fit and transform the texts
    tfidf_inclusion = vectorizer_inclusion.fit_transform(inclusion_criteria)
    tfidf_exclusion = vectorizer_exclusion.fit_transform(exclusion_criteria)
    tfidf_title_desc_mesh = vectorizer_title_desc_mesh.fit_transform(title_desc_mesh)

    # Create inverted indexes
    inclusion_index = create_inverted_index(vectorizer_inclusion.get_feature_names_out(), tfidf_inclusion, doc_ids)
    exclusion_index = create_inverted_index(vectorizer_exclusion.get_feature_names_out(), tfidf_exclusion, doc_ids)
    title_desc_mesh_index = create_inverted_index(vectorizer_title_desc_mesh.get_feature_names_out(), tfidf_title_desc_mesh, doc_ids)

    # Save to JSON files
    with open('inclusion_criteria_index.json', 'w') as f:
        json.dump(inclusion_index, f)
    with open('exclusion_criteria_index.json', 'w') as f:
        json.dump(exclusion_index, f)
    with open('title_desc_mesh_index.json', 'w') as f:
        json.dump(title_desc_mesh_index, f)
