# файл с классификацией

import os
from ultralytics import YOLO
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import yaml
import torch

class SkinCancerDetector:
    def __init__(self, base_dir):
        """Инициализация с базовой директорией проекта."""
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / 'dataset'
        self.processed_dir = self.base_dir / 'processed_dataset'
        self.model_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        # Создаем необходимые директории
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
    def create_dataset_yaml(self):
        """Создает YAML конфигурацию датасета."""
        yaml_path = self.processed_dir / 'dataset.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data_yaml = {
            'path': '.',
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': 7,
            'names': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        }

        # Записываем данные в файл dataset.yaml
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        print(f"++++++++++++++++++++++++++++++++++++++++++++++++{yaml_path}")
        return yaml_path

    def prepare_folders(self):
        """Подготовка структуры папок для обработанного датасета."""
        splits = ['train', 'val', 'test']
        classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        for split in splits:
            for cls in classes:
                (self.processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

        # Отладочный вывод для проверки структуры папок
        print(f"Processed dataset structure created at: {self.processed_dir}")
        print(f"Contents: {list(self.processed_dir.iterdir())}")

        return splits, classes

    def process_dataset(self):
        """Обработка и разделение датасета."""
        print("Processing dataset...")

        # Чтение CSV с разметкой
        annot = pd.read_csv(self.dataset_dir / 'GroundTruth.csv')

        # Разделение данных
        train_val, test = train_test_split(
            annot,
            test_size=0.1,
            stratify=annot[annot.columns[1:]].idxmax(axis=1),
            random_state=42
        )
        train, val = train_test_split(
            train_val,
            test_size=0.2,
            stratify=train_val[train_val.columns[1:]].idxmax(axis=1),
            random_state=42
        )

        # Распределение изображений по папкам
        splits_data = {'train': train, 'val': val, 'test': test}
        
        for split_name, split_data in splits_data.items():
            print(f"Processing {split_name} split...")
            for cls in annot.columns[1:]:
                class_images = split_data[split_data[cls] == 1]['image'].tolist()
                for img in class_images:
                    src = self.dataset_dir / 'images' / f'{img}.jpg'
                    dst = self.processed_dir / split_name / cls / f'{img}.jpg'
                    if src.exists():
                        shutil.copy(src, dst)
                    else:
                        print(f"Warning: Image not found {src}")

        return {'train_size': len(train), 'val_size': len(val), 'test_size': len(test)}

    def train_model(self, yaml_path):
        """Обучение модели YOLO."""
        print("Initializing YOLO model training...")

        # Проверка существования папок
        for split in ['train', 'val', 'test']:
            split_path = self.processed_dir / split
            if not split_path.exists() or not any(split_path.iterdir()):
                raise FileNotFoundError(f"Directory not found or empty: {split_path}")

        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = YOLO("yolo11s-cls.pt")

        results = model.train(
            data=str(yaml_path),  # Путь к YAML
            epochs=300,
            imgsz=224,
            device=device,
            batch=32,
            patience=50,
            save=True,
            project=str(self.results_dir),
            name="skin_cancer_model_3",
            exist_ok=True
        )

        metrics = model.val()
        export_path = model.export(format="onnx", save=True)

        return model, metrics, export_path

    def run(self):
        """Основной метод запуска всего процесса."""
        try:
            print("Starting skin cancer detection project...")
            self.prepare_folders()
            dataset_stats = self.process_dataset()
            
            print("Dataset statistics:")
            for split, size in dataset_stats.items():
                print(f"{split}: {size} images")

            yaml_path = self.create_dataset_yaml()
            print(f"Created dataset configuration at: {yaml_path}")

            model, metrics, export_path = self.train_model("/home/user/MoleScane/processed_dataset")
            print(f"Model training completed. Exported model saved to: {export_path}")

            print("Project execution completed successfully!")
            return model, metrics

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise

def main():
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    detector = SkinCancerDetector(project_dir)
    model, metrics = detector.run()
    
    print("\nTraining metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
