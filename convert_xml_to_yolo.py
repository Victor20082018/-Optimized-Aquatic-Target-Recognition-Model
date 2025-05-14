import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Define paths
base_path = Path('datasets/water_surface')
annotations_dir = base_path / 'Annotations'
images_dir = base_path / 'JPEGImages'
output_labels_dir = base_path / 'labels'

# Create output directory if it doesn't exist
os.makedirs(output_labels_dir, exist_ok=True)

# Class names and their corresponding IDs
class_names = {
    'fanchuan': 0,
    'person': 1,
    'otherboat': 2,
    'bird': 3,
    'youlun': 4,
    'warship': 5,
    'yacht': 6,
    'airplane': 7,
    'Cargoship': 8,
    'fish': 9
}

def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    label_lines = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')

        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Convert bounding box to YOLO format (center_x, center_y, width, height)
        center_x = (xmin + xmax) / 2.0 / image_width
        center_y = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        class_id = class_names[class_name]
        label_line = f"{class_id} {center_x} {center_y} {width} {height}\n"
        label_lines.append(label_line)

    return label_lines

for annotation_file in annotations_dir.glob('*.xml'):
    image_name = annotation_file.stem + '.jpg'  # Assuming images are in JPEG format
    label_file = output_labels_dir / (annotation_file.stem + '.txt')

    label_lines = convert_annotation(annotation_file)

    with open(label_file, 'w') as f:
        f.writelines(label_lines)

print("Conversion completed.")