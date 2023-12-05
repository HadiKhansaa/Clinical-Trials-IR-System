import xml.etree.ElementTree as ET
import json
from create_embeddings_BERT import create_bert_embeddings
from transformers import BertTokenizer, BertModel

def extract_topics_from_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    topics = []
    for topic in root.findall('./topic'):
        topic_text = topic.text.strip()
        topics.append(topic_text)

    return topics

if __name__ == "__main__":
    # Extract topics from the XML file
    topics = extract_topics_from_xml('topics.xml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Generate embeddings for each topic
    topic_embeddings = create_bert_embeddings(topics, tokenizer, model)

    # Saving the topic embeddings to a JSON file
    topic_embeddings_json = {f"topic_{i+1}": emb for i, emb in enumerate(topic_embeddings)}
    with open('topic_embeddings.json', 'w') as f:
        json.dump(topic_embeddings_json, f)
