import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
import torch
from torch.utils.data import DataLoader, Dataset
from super_gradients.training import Trainer, models
from super_gradients.common.object_names import Models
import matplotlib.pyplot as plt

# Пути к данным и директориям
CSV_PATH = "/home/user/MoleScane/dataset/GroundTruth.csv"
IMAGES_DIR = "/home/user/MoleScane/dataset/images/"
MASKS_DIR = "/home/user/MoleScane/dataset/masks/"
LABELS_DIR = "/home/user/MoleScane/dataset/labels/"
MODEL_DIR = "/home/user/MoleScane/results/yolo_nas_model/"
WEIGHTS_DIR = os.path.join(MODEL_DIR, "weights/")
GRAPHS_DIR = os.path.join(MODEL_DIR, "graphs/")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "yolo_nas_checkpoint.pth")

# Создаём необходимые директории
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Словарь классов с ID
CLASS_MAPPING = {
    "MEL": 0, "NV": 1, "BCC": 2, 
    "AKIEC": 3, "BKL": 4, "DF": 5, "VASC": 6
}

# Загружаем CSV с метками классов
data = pd.read_csv(CSV_PATH)

# Функция для конвертации масок в YOLO-формат
def mask_to_yolo_bbox(mask):
    coords = np.argwhere(mask == 1)
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    x_center = (x_min + x_max) / 2 / mask.shape[1]
    y_center = (y_min + y_max) / 2 / mask.shape[0]
    width = (x_max - x_min) / mask.shape[1]
    height = (y_max - y_min) / mask.shape[0]

    return x_center, y_center, width, height

# Генерация аннотаций
for _, row in data.iterrows():
    image_name = row['image']
    mask_path = os.path.join(MASKS_DIR, f"{image_name}_segmentation.png")

    if not os.path.exists(mask_path):
        print(f"Пропущено: Маска для {image_name} не найдена.")
        continue

    mask = np.array(Image.open(mask_path).convert('L'))
    mask = (mask > 128).astype(np.uint8)

    bbox = mask_to_yolo_bbox(mask)
    if bbox is None:
        print(f"Пропущено: Пустая маска для {image_name}.")
        continue

    x_center, y_center, width, height = bbox

    for class_name, value in row.items():
        if class_name != 'image' and value == 1.0:
            class_id = CLASS_MAPPING[class_name]
            break

    label_file = os.path.join(LABELS_DIR, f"{image_name}.txt")
    with open(label_file, 'w') as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Разделение данных на train, val и test
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

print(f"Тренировочных: {len(train_data)}, Валидационных: {len(val_data)}, Тестовых: {len(test_data)}")

# Класс Dataset для YOLO
class YOLODataset(Dataset):
    def __init__(self, data, img_dir, label_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image']}.jpg")
        label_path = os.path.join(self.label_dir, f"{row['image']}.txt")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            label_data = f.readline().strip().split()
        class_id = int(label_data[0])
        bbox = np.array(label_data[1:], dtype=np.float32)

        return image, class_id, bbox

# Создание DataLoader для train, val и test
train_dataset = YOLODataset(train_data, IMAGES_DIR, LABELS_DIR)
val_dataset = YOLODataset(val_data, IMAGES_DIR, LABELS_DIR)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# Создаём тренера
trainer = Trainer(experiment_name="moles_classification")

# Загружаем предобученную модель YOLO-NAS
model = models.get(Models.YOLO_NAS_M, pretrained_weights=None, num_classes=7)

# Параметры обучения
train_params = {
    "max_epochs": 50,
    "lr_mode": "cosine",
    "optimizer": "Adam",
    "initial_lr": 0.001,
}

# Запуск обучения
trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_loader,
    valid_loader=val_loader,
)

# Сохранение модели
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"Модель сохранена по пути: {CHECKPOINT_PATH}")
