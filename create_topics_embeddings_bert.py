import xml.etree.ElementTree as ET
import json
from create_embeddings_BERT import create_bert_embeddings
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

def extract_topics_from_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    topics = []
    for topic in root.findall('./topic'):
        topic_text = topic.text.strip()
        topics.append(topic_text)
    print(topics)
    return topics

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

if __name__ == "__main__":
    # Extract topics from the XML file
    # topics = extract_topics_from_xml('topics.xml')
    with open("topic.txt",'r') as f :
        topic = f.readline()
    tokenizer3 = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model3 = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')
    # tokenizer2 = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    # model2 = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2')

    for TOPIC_ID in range(1,11):
        topic = load_topic(TOPIC_ID, "topics.xml")
        # # Generate embeddings for each topic
        # topic_embeddings = create_bert_embeddings(topic, tokenizer, model)

        # # Saving the topic embeddings to a JSON file
        # with open('files_query10\\topic_embeddings_BERT.json', 'w') as f:
        #     json.dump(topic_embeddings, f)

        # # Generate embeddings for each topic
        # topic_embeddings = create_bert_embeddings(topic, tokenizer2, model2)

        # # Saving the topic embeddings to a JSON file
        # with open('files_query10\\topic_embeddings_bioBERT.json', 'w') as f:
        #     json.dump(topic_embeddings, f)

        # Generate embeddings for each topic
        topic_embeddings = create_bert_embeddings(topic, tokenizer3, model3)
        # print(topic)
        # Saving the topic embeddings to a JSON file
        with open(f'files_query{TOPIC_ID}\\topic_embeddings_Clinical.json', 'w') as f:
            json.dump(topic_embeddings, f)
