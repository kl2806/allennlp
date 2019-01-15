import csv
import json

target_file = open("quarel-preamble-v1-train.json", "w")

with open("quarel-v1-train.csv", "r") as relations:
    with open("./allennlp/tests/fixtures/data/quarel/quarel-v1-train.json") as source_file:
        reader = csv.reader(relations)
        for relation, source_line in zip(reader, source_file):
            json_line = json.loads(source_line)
            json_line['property_1'] = relation[1]
            json_line['property_2'] = relation[2]
            json_line['relation'] = relation[3]
            
            assert relation[0] == json_line['id']

            if relation[3] == '1':
                preamble = f"If {relation[1]} is higher, then {relation[2]} is higher. "
            else:
                preamble = f"If {relation[1]} is higher, then {relation[2]} is lower. "
            json_line['question'] = preamble + json_line['question']
            json.dump(json_line, target_file)
            target_file.write('\n')
