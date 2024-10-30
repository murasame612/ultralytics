import json
import os

def convert(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + '.txt')
    with open(output_file_path, 'w') as out_file:
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])

            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            out_file.write(f"0 {x_center} {y_center} {width} {height}\n")

json_dir = ''
output_dir = ''
os.makedirs(output_dir, exist_ok=True)

for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        convert(os.path.join(json_dir, json_file), output_dir)

