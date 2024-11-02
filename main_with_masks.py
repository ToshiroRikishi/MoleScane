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
        """Создает YAML конфигурацию датасета для сегментации."""
        yaml_path = self.processed_dir / 'dataset.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data_yaml = {
            'path': str(self.processed_dir),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': 7,
            'names': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        }

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        print(f"Создан файл конфигурации датасета: {yaml_path}")
        return yaml_path

    def prepare_folders(self):
        """Подготовка структуры папок для обработанного датасета."""
        for split in ['train', 'val', 'test']:
            for cls in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']:
                (self.processed_dir / split / cls).mkdir(parents=True, exist_ok=True)
        print(f"Структура обработанного датасета создана в: {self.processed_dir}")

    def process_dataset(self):
        """Обработка и разделение датасета, включая маски."""
        print("Начата обработка датасета...")

        annot = pd.read_csv(self.dataset_dir / 'GroundTruth.csv')
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
            print(f"Обработка части: {split_name}")
            for _, row in split_data.iterrows():
                img_name = row['image']
                cls_name = row[1:].idxmax()
                src_img = self.dataset_dir / 'images' / cls_name / f'{img_name}.jpg'
                dst_img = self.processed_dir / split_name / cls_name / f'{img_name}.jpg'
                
                src_mask = self.dataset_dir / 'masks' / cls_name / f'{img_name}_segmentation.png'
                dst_mask = self.processed_dir / split_name / cls_name / f'{img_name}.png'

                if src_img.exists():
                    shutil.copy(src_img, dst_img)
                if src_mask.exists():
                    shutil.copy(src_mask, dst_mask)
                else:
                    print(f"Предупреждение: Маска для {src_mask} не найдена")

        return {'train_size': len(train), 'val_size': len(val), 'test_size': len(test)}

    def train_model(self, yaml_path):
        """Обучение модели YOLO для сегментации."""
        print("Запуск обучения модели YOLO...")
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Используемое устройство: {device}")

        model = YOLO("yolo11n-seg.pt")

        results = model.train(
            data=str(yaml_path), 
            epochs=300,
            imgsz=640,
            device=device,
            batch=32,
            patience=50,
            save=True,
            project=str(self.results_dir),
            name="skin_cancer_segmentation",
            exist_ok=True
        )

        metrics = model.val()
        export_path = model.export(format="onnx", save=True)

        return model, metrics, export_path

    def run(self):
        """Основной метод запуска всего процесса."""
        try:
            print("Запуск проекта по сегментации кожных заболеваний...")
            self.prepare_folders()
            dataset_stats = self.process_dataset()
            
            print("Статистика датасета:")
            for split, size in dataset_stats.items():
                print(f"{split}: {size} изображений")

            yaml_path = self.create_dataset_yaml()
            print(f"Конфигурация датасета создана: {yaml_path}")

            model, metrics, export_path = self.train_model(yaml_path)
            print(f"Обучение завершено. Экспортированная модель сохранена в: {export_path}")

            print("Проект успешно завершен!")
            return model, metrics

        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
            raise

def main():
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    detector = SkinCancerDetector(project_dir)
    model, metrics = detector.run()
    
    print("\nМетрики обучения:")
    print(metrics)

if __name__ == "__main__":
    main()
