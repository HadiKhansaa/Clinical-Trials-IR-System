import nltk
import os
import re
import json
import math
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import spacy

# nltk.download('punkt')


def tokenize_and_stem(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalpha()]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def extract_components(content):
    inc_match = re.search(r'<textblock>.*?Inclusion Criteria:(.*?)Exclusion Criteria:(.*?)</textblock>', content, re.DOTALL)
    title_match = re.search(r'<brief_title>(.*?)</brief_title>', content)
    summary_match = re.search(r'<brief_summary>.*?<textblock>(.*?)</textblock>.*?</brief_summary>', content, re.DOTALL)
    # desc_match = re.search(r'<detailed_description>.*?<textblock>(.*?)</textblock>.*?</detailed_description>', content, re.DOTALL)
    mesh_match = re.findall(r'<mesh_term>(.*?)</mesh_term>', content)
    inc_criteria = inc_match.groups()[0] if inc_match else ''
    exc_criteria = inc_match.groups()[1] if inc_match else ''
    if exc_criteria == '':
        inc_match = re.search(r'<textblock>.*?Inclusion Criteria:(.*?)</textblock>', content, re.DOTALL)
        inc_criteria = inc_match.groups()[0] if inc_match else ''
    title = title_match.group(1) if title_match else ''
    # desc = desc_match.group(1) if desc_match else ''
    summary = summary_match.group(1) if summary_match else ''
    mesh_terms = ' '.join(mesh_match)

    return inc_criteria, exc_criteria, f"{title} {summary} {mesh_terms}"

def calculate_bm25(doc_length, avg_doc_length, term_freq, num_docs, doc_freq, k1=1.5, b=0.75):
    idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    tf = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_length / avg_doc_length))
    return tf * idf

def create_inverted_index(corpus, doc_ids):
    doc_freq = {}
    doc_lengths = []
    for doc in corpus:
        doc_lengths.append(len(doc))
        counts = Counter(doc)
        for term in counts:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    num_docs = len(corpus)
    index = {}

    for i, doc in enumerate(corpus):
        for term in set(doc):
            term_freq = doc.count(term)
            score = calculate_bm25(len(doc), avg_doc_length, term_freq, num_docs, doc_freq[term])
            if term not in index:
                index[term] = {}
            index[term][doc_ids[i]] = score

    return index

nlp = spacy.load("en_ner_bc5cdr_md")

def filter_medical_terms(text):
    doc = nlp(text)
    # print(list(doc.ents))
    # Filter out entities that are identified as medical terms
    medical_terms = [ent.text for ent in doc.ents]
    return medical_terms

if __name__ == "__main__":
    inclusion_criteria = []
    exclusion_criteria = []
    title_desc_mesh = []
    doc_ids = []

    # PATH_TO_TRIALS = 'C:\\Users\\Hp\\Documents\\CMPS M\\CMPS 365\\project\\trecs\\trials'
    PATH_TO_TRIALS = 'trials_query1'

    # get documents with relevence feedback
    with open('documents.json', 'r') as file:
        docs = json.load(file)
    documents = set(docs)

    i = 1
    for file in os.listdir(os.path.join(PATH_TO_TRIALS)):
        with open(os.path.join(PATH_TO_TRIALS, file), 'r') as f:
            if file.endswith('.xml') and file[:-4] in documents:
                try:
                    content = f.read()
                except:
                    continue
                inc, exc, tdm = extract_components(content)
                if inc !='':
                    # inclusion_criteria.append(' '.join(filter_medical_terms(inc)))
                    inclusion_criteria.append(inc)
                if exc != '':
                    exclusion_criteria.append(exc)
                    # exclusion_criteria.append(' '.join(filter_medical_terms(exc)))
                if tdm != '':
                    title_desc_mesh.append(tdm)
                    # title_desc_mesh.append(' '.join(filter_medical_terms(tdm)))
                doc_ids.append(file[:-4])  # Assuming file names are the document IDs
                print(f"Processed {i} files")  # Print progress
                i += 1

    # Tokenize and stem
    tokenized_inclusion = [tokenize_and_stem(text) for text in inclusion_criteria]
    tokenized_exclusion = [tokenize_and_stem(text) for text in exclusion_criteria]
    tokenized_title_desc_mesh = [tokenize_and_stem(text) for text in title_desc_mesh]

    # Create inverted indexes
    inclusion_index = create_inverted_index(tokenized_inclusion, doc_ids)
    exclusion_index = create_inverted_index(tokenized_exclusion, doc_ids)
    title_desc_mesh_index = create_inverted_index(tokenized_title_desc_mesh, doc_ids)

    # Save to JSON files
    with open('inclusion_criteria_index_bm25.json', 'w') as f:
        json.dump(inclusion_index, f)
    with open('exclusion_criteria_index_bm25.json', 'w') as f:
        json.dump(exclusion_index, f)
    with open('title_desc_mesh_index_bm25.json', 'w') as f:
        json.dump(title_desc_mesh_index, f)
