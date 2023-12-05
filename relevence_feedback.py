import json

with open('relevence_feedback.txt', 'r') as file:
    data = file.read()
# Parsing the data into a dictionary
relevance_feedback = {}
for line in data.strip().split("\n"):
    parts = line.split()
    if len(parts) == 4:
        query_id, _, document_id, relevance = parts
        relevance_feedback[f"{query_id}_{document_id}"] = int(relevance)

# File path for JSON file
json_file_path = 'relevance_feedback.json'

# Writing the dictionary to a JSON file
with open(json_file_path, 'w') as file:
    json.dump(relevance_feedback, file)

json_file_path

