import json

with open('relevance_feedback.json', 'r') as file:
    relevance_feedback = json.load(file)

# count the number of unique documents
#adding only query 1 documents
unique_documents = set()
for key in relevance_feedback.keys():
    if key.split('_')[0] == '6':
        unique_documents.add(key.split('_')[1])

print(f"Number of unique documents: {len(unique_documents)}")

with open('documents_q6.json', 'w') as file:
    json.dump(sorted(list(unique_documents)), file)