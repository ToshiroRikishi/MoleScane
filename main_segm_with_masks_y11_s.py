# файл сегментации

import os
from ultralytics import YOLO
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import yaml
import torch
import cv2
import numpy as np

class SkinCancerSegmentation:
    def __init__(self, base_dir):
        """Инициализация с базовой директорией проекта."""
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / 'dataset'
        self.processed_dir = self.base_dir / 'processed_dataset'
        self.model_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        # Создаем необходимые директории
        for dir_path in [self.model_dir, self.results_dir, self.processed_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def create_dataset_yaml(self):
        """Создает YAML конфигурацию датасета для сегментации."""
        yaml_path = self.processed_dir / 'dataset.yaml'
        
        data_yaml = {
            'path': str(self.processed_dir.absolute()),  # Абсолютный путь к датасету
            'train': 'train/images',  # Путь к тренировочным изображениям
            'val': 'val/images',      # Путь к валидационным изображениям
            'test': 'test/images',    # Путь к тестовым изображениям
            'names': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
            'nc': 7,  # Количество классов
        }

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        return yaml_path

    def prepare_yolo_folders(self):
        """Подготовка структуры папок для YOLO сегментации."""
        splits = ['train', 'val', 'test']
        for split in splits:
            # Создаем папки для изображений и меток
            (self.processed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def convert_mask_to_yolo(self, mask, image_size):
        """Конвертация бинарной маски в формат YOLO для сегментации."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Берем самый большой контур
        contour = max(contours, key=cv2.contourArea)
        
        # Нормализуем координаты
        points = contour.reshape(-1, 2)
        points = points.astype(float)
        points[:, 0] /= image_size[1]  # normalize x by width
        points[:, 1] /= image_size[0]  # normalize y by height
        
        return points

    def process_dataset(self):
        """Обработка и разделение датасета с учетом сегментации."""
        print("Processing dataset for segmentation...")

        # Чтение CSV с разметкой
        annot = pd.read_csv(self.dataset_dir / 'GroundTruth.csv')
        class_columns = annot.columns[1:]  # Получаем названия столбцов классов

        # Разделение данных
        train_val, test = train_test_split(
            annot, test_size=0.1,
            stratify=annot[class_columns].idxmax(axis=1),
            random_state=42
        )
        train, val = train_test_split(
            train_val, test_size=0.2,
            stratify=train_val[class_columns].idxmax(axis=1),
            random_state=42
        )

        splits_data = {'train': train, 'val': val, 'test': test}
        
        for split_name, split_data in splits_data.items():
            print(f"Processing {split_name} split...")
            
            for idx, row in split_data.iterrows():
                img_id = row['image']
                # Получаем индекс класса, используя только столбцы классов
                class_values = [row[col] for col in class_columns]
                class_idx = class_values.index(1)  # Находим индекс класса, где значение 1
                
                # Копируем изображение
                src_img = self.dataset_dir / 'images' / f'{img_id}.jpg'
                dst_img = self.processed_dir / split_name / 'images' / f'{img_id}.jpg'
                
                if src_img.exists():
                    shutil.copy(src_img, dst_img)
                    
                    # Обрабатываем маску
                    mask_path = self.dataset_dir / 'masks' / f'{img_id}_segmentation.png'
                    if mask_path.exists():
                        # Читаем маску и изображение
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        img = cv2.imread(str(src_img))
                        h, w = img.shape[:2]
                        
                        # Конвертируем маску в формат YOLO
                        yolo_segments = self.convert_mask_to_yolo(mask, (h, w))
                        
                        if yolo_segments is not None:
                            # Создаем файл с разметкой в формате YOLO
                            label_path = self.processed_dir / split_name / 'labels' / f'{img_id}.txt'
                            with open(label_path, 'w') as f:
                                # Записываем: class_idx x1 y1 x2 y2 ... xn yn
                                points_flat = yolo_segments.reshape(-1)
                                points_str = ' '.join(map(str, points_flat))
                                f.write(f"{class_idx} {points_str}\n")
                    else:
                        print(f"Warning: Mask not found for image {img_id}")
                else:
                    print(f"Warning: Image not found {src_img}")

        return {'train_size': len(train), 'val_size': len(val), 'test_size': len(test)}

    def train_model(self, yaml_path):
        """Обучение модели YOLO для сегментации."""
        print("Initializing YOLO segmentation model training...")

        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Используем модель с сегментацией
        model = YOLO("yolo11s-seg.pt")

        results = model.train(
            data=str(yaml_path),
            epochs=10,
            imgsz=640,  # Увеличенный размер для лучшей сегментации
            device=device,
            batch=16,   # Уменьшенный размер батча из-за сегментации
            patience=50,
            save=True,
            project=str(self.results_dir),
            name="skin_cancer_segmentation",
            exist_ok=True
        )

        # Валидация модели
        metrics = model.val()
        
        # Экспорт модели
        export_path = model.export(format="onnx", save=True)

        return model, metrics, export_path

    def run(self):
        """Основной метод запуска всего процесса."""
        try:
            print("Starting skin cancer segmentation project...")
            self.prepare_yolo_folders()
            dataset_stats = self.process_dataset()
            
            print("Dataset statistics:")
            for split, size in dataset_stats.items():
                print(f"{split}: {size} images")

            yaml_path = self.create_dataset_yaml()
            print(f"Created dataset configuration at: {yaml_path}")

            model, metrics, export_path = self.train_model(yaml_path)
            print(f"Model training completed. Exported model saved to: {export_path}")

            return model, metrics

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise

def main():
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    segmentation = SkinCancerSegmentation(project_dir)
    model, metrics = segmentation.run()
    
    print("\nTraining metrics:")
    print(f"Segmentation mAP: {metrics.seg.map}")
    print(f"Box mAP: {metrics.box.map}")

if __name__ == "__main__":
    main()