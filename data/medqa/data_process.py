import json

with open("demonstrations/train.json", 'r') as f:
    data = [json.loads(line) for line in f]

final_data = []
for index, item in enumerate(data):
    id = item['id']
    question = item['sent1'] + "\nAnswer Choices: (A) " + item['ending0'] + " (B) " + item['ending1'] + " (C) " + item['ending2'] + " (D) " + item['ending3']
    label = item['label']
    if label == 0:
        label = ["A"]
    elif label == 1:
        label = ["B"]
    elif label == 2:
        label = ["C"]
    elif label == 3:
        label = ["D"]
    else:
        raise ValueError("Invalid label")
    final_data.append({"id": id, "question": question, "label": label})

output_path = "demonstrations/medqa_input_text.jsonl"

with open(output_path, 'w') as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")