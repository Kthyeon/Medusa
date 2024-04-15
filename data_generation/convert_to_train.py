# import json

# with open('self_vicuna_0.3_seed1.json', 'r') as file:
#     data = file.read().strip()

# # 각 JSON 객체가 새 줄에 구분되어 있다고 가정
# json_objects = data.split('\n')

# all_objects = []
# for obj in json_objects:
#     try:
#         # JSON 문자열을 파이썬 딕셔너리로 변환
#         parsed_json = json.loads(obj)
#         all_objects.append(parsed_json)
#     except json.JSONDecodeError as e:
#         print(obj)
#         print(f"JSON parsing_error: {e}")

# # 모든 JSON 객체를 하나의 리스트로 저장
# with open('self_vicuna_0.3_reformat.json', 'w') as output_file:
#     json.dump(all_objects, output_file)

import json

def reformat_json(input_file, output_file):
    # Load the JSON data from the input file
    with open(input_file, 'r') as file:
        data = file.read().strip()

    json_objects = data.split('\n')
    # Transform the structure as needed
    transformed_data = []
    for obj in json_objects:
        obj = json.loads(obj)
        new_item = {
            "id": obj['id'],
            "conversations": []
        }
        for item in obj['conversations']:
            # Assuming you want to maintain the id and create a new format for each conversation
            new_item['conversations'].append(
                    {
                        "from": item['from'],
                        "value": item['value']
                    })
        transformed_data.append(new_item)
    # Save the transformed data to the output file
    with open(output_file, 'w') as file:
        json.dump(transformed_data, file, indent=4)

    print("JSON data has been reformatted and saved to", output_file)

# Set the input and output file paths
input_path = 'self_vicuna_0.3_seed1.json'
output_path = 'self_vicuna_0.3_reformat.json'

# Call the function to reformat the JSON
reformat_json(input_path, output_path)
