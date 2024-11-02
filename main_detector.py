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
            'path': str(self.processed_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 7,
            'names': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        }

        # Записываем данные в файл dataset.yaml
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        return yaml_path

    def prepare_folders(self):
        """Подготовка структуры папок для обработанного датасета."""
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.processed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        print(f"Processed dataset structure created at: {self.processed_dir}")
        return splits

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

        splits_data = {'train': train, 'val': val, 'test': test}
        
        for split_name, split_data in splits_data.items():
            print(f"Processing {split_name} split...")
            for index, row in split_data.iterrows():
                img_name = row['image']
                src_image = self.dataset_dir / 'images' / f'{img_name}.jpg'
                dst_image = self.processed_dir / split_name / 'images' / f'{img_name}.jpg'
                
                # Перемещение изображений
                if src_image.exists():
                    shutil.copy(src_image, dst_image)
                else:
                    print(f"Warning: Image not found {src_image}")
                
                # Перемещение аннотаций
                src_label = self.dataset_dir / 'labels' / f'{img_name}.txt'
                dst_label = self.processed_dir / split_name / 'labels' / f'{img_name}.txt'
                if src_label.exists():
                    shutil.copy(src_label, dst_label)
                else:
                    print(f"Warning: Label not found {src_label}")

        return {'train_size': len(train), 'val_size': len(val), 'test_size': len(test)}

    def train_model(self, yaml_path):
        """Обучение модели YOLO для детекции объектов."""
        print("Initializing YOLO model training...")

        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = YOLO("yolo11n.pt")  # или yolo11s для другой модели

        results = model.train(
            data=str(yaml_path),
            epochs=100,
            imgsz=640,
            device=device,
            batch=16,
            patience=10,
            save=True,
            project=str(self.results_dir),
            name="skin_cancer_detection",
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

            model, metrics, export_path = self.train_model(yaml_path)
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
