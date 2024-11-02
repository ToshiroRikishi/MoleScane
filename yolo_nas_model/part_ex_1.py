import os
import shutil
import random
from tqdm import tqdm
import cv2
import json
import numpy as np

# Пути к данным
base_dir = "/home/user/MoleScane/dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Папки для train и valid
train_images_dir = os.path.join(images_dir, "train")
valid_images_dir = os.path.join(images_dir, "valid")
train_labels_dir = os.path.join(labels_dir, "train")
valid_labels_dir = os.path.join(labels_dir, "valid")

# Создание директорий для train и valid
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)

# Получение списка файлов изображений и разделение 80/20
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
random.shuffle(image_files)
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
valid_files = image_files[split_index:]

# Копирование изображений и меток
for file_name in train_files:
    shutil.copy2(os.path.join(images_dir, file_name), train_images_dir)
    label_file = file_name.replace(".jpg", ".txt")
    shutil.copy2(os.path.join(labels_dir, label_file), train_labels_dir)

for file_name in valid_files:
    shutil.copy2(os.path.join(images_dir, file_name), valid_images_dir)
    label_file = file_name.replace(".jpg", ".txt")
    shutil.copy2(os.path.join(labels_dir, label_file), valid_labels_dir)

# Определение классов
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NUM_CLASSES = len(CLASS_NAMES)

# Функция для конвертации в COCO формат
def convert_to_coco(image_dir, label_dir, output_annotation_json):
    coco_annotation = {
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": name, "name": name, "id": idx} for idx, name in enumerate(CLASS_NAMES)],
    }
    annotation_id = 1  # Уникальный ID для аннотации

    for image_file in tqdm(os.listdir(image_dir)):
        if image_file.endswith(".jpg"):
            image_id = int(image_file.split("_")[1].split(".")[0])  # Извлечение ID из имени файла
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # Добавление информации об изображении в аннотацию COCO
            coco_annotation["images"].append({
                "file_name": image_file,
                "height": height,
                "width": width,
                "id": image_id
            })

            # Чтение аннотаций
            label_file = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        category_id = int(parts[0])
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                        # Конвертация координат в COCO формат (x, y, width, height)
                        x = int((x_center - bbox_width / 2) * width)
                        y = int((y_center - bbox_height / 2) * height)
                        bbox_width = int(bbox_width * width)
                        bbox_height = int(bbox_height * height)

                        # Добавление аннотации для объекта
                        coco_annotation["annotations"].append({
                            "id": annotation_id,
                            "category_id": category_id,
                            "image_id": image_id,
                            "bbox": [x, y, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0
                        })
                        annotation_id += 1

    # Сохранение аннотаций в JSON файл
    with open(output_annotation_json, "w") as f:
        json.dump(coco_annotation, f, indent=4)

# Генерация файлов аннотаций для train и valid
convert_to_coco(train_images_dir, train_labels_dir, "/home/user/MoleScane/dataset/train_annotations.coco.json")
convert_to_coco(valid_images_dir, valid_labels_dir, "/home/user/MoleScane/dataset/valid_annotations.coco.json")
