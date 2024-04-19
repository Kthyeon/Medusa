import json

def load_and_transform_json(input_file):
    # Load and parse the JSON data from the input file
    with open(input_file, 'r') as file:
        json_objects = file.read().strip().split('\n')

    transformed_data = []
    for obj in json_objects:
        obj = json.loads(obj)
        new_item = {
            "id": obj['id'],
            "conversations": [{"from": conv['from'], "value": conv['value']} for conv in obj['conversations']]
        }
        transformed_data.append(new_item)
    return transformed_data

def reformat_json(input_files, output_file):
    transformed_data = []
    # Process each input file
    for input_file in input_files:
        transformed_data.extend(load_and_transform_json(input_file))

    # Save the combined and transformed data to the output file
    with open(output_file, 'w') as file:
        json.dump(transformed_data, file, indent=4)

    print("JSON data has been reformatted and saved to", output_file)

# Set the input and output file paths
input_files = ['self_vicuna_0.1_seed4.json', 'self_vicuna_0.1_seed5.json', 'self_vicuna_0.2_seed4.json', 'self_vicuna_0.2_seed5.json', 
    'self_vicuna_0.4_seed2.json', 'self_vicuna_0.4_seed3.json', 'self_vicuna_0.4_seed4.json',
    'self_vicuna_0.5_seed2.json', 'self_vicuna_0.5_seed3.json', 'self_vicuna_0.5_seed4.json', 
    'self_vicuna_0.6_seed2.json']
output_path = 'self_vicuna_1.5b_tokens_reformat.json'

# Call the function to reformat the JSON
reformat_json(input_files, output_path)
