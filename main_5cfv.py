import os
from ultralytics import YOLO
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import shutil
from pathlib import Path
import yaml
import torch

class SkinCancerDetectorCV:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / 'dataset'
        self.processed_dir = self.base_dir / 'processed_dataset'
        self.model_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    def create_dataset_yaml(self, fold_idx):
        yaml_path = self.processed_dir / f'dataset_fold_{fold_idx}.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data_yaml = {
            'path': '.',
            'train': f'train_fold_{fold_idx}',
            'val': f'val_fold_{fold_idx}',
            'nc': 7,
            'names': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        }

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        return yaml_path

    def prepare_folders(self, fold_idx):
        for split in [f'train_fold_{fold_idx}', f'val_fold_{fold_idx}']:
            for cls in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']:
                (self.processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def process_dataset(self):
        annot = pd.read_csv(self.dataset_dir / 'GroundTruth.csv')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        best_model = None
        best_top1 = float('-inf')
        best_fold = None

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(annot, annot[annot.columns[1:]].idxmax(axis=1))):
            print(f"Processing fold {fold_idx + 1}...")
            train_data = annot.iloc[train_idx]
            val_data = annot.iloc[val_idx]
            
            self.prepare_folders(fold_idx)

            for split_name, split_data in zip([f'train_fold_{fold_idx}', f'val_fold_{fold_idx}'], [train_data, val_data]):
                for cls in annot.columns[1:]:
                    class_images = split_data[split_data[cls] == 1]['image'].tolist()
                    for img in class_images:
                        src = self.dataset_dir / 'images' / f'{img}.jpg'
                        dst = self.processed_dir / split_name / cls / f'{img}.jpg'
                        if src.exists():
                            shutil.copy(src, dst)
                        else:
                            print(f"Warning: Image not found {src}")

            yaml_path = self.create_dataset_yaml(fold_idx)
            model, metrics, export_path = self.train_model("/home/user/MoleScane/processed_dataset")

            current_top1 = metrics.top1
            if current_top1 > best_top1:
                best_top1 = current_top1
                best_model = export_path
                best_fold = fold_idx + 1

            print(f"Fold {fold_idx + 1} training completed with top1 accuracy: {current_top1}")

        print(f"Best model saved at: {best_model} with top1 accuracy: {best_top1} from fold {best_fold}")
        return best_model

    def train_model(self, yaml_path):
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = YOLO("yolo11s-cls.pt")

        results = model.train(
            data=str(yaml_path),
            epochs=1,
            imgsz=224,
            device=device,
            batch=32,
            patience=50,
            save=True,
            project=str(self.results_dir),
            name="skin_cancer_model_5_fcv",
            exist_ok=True
        )

        metrics = model.val()
        export_path = model.export(format="onnx", save=True)

        return model, metrics, export_path

    def run(self):
        try:
            print("Starting 5-fold cross-validation...")
            best_model = self.process_dataset()

            print("5-fold cross-validation completed successfully!")
            print(f"Best model path: {best_model}")

            return best_model

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise

def main():
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    detector = SkinCancerDetectorCV(project_dir)
    best_model = detector.run()
    
    print("\nBest model saved at:")
    print(best_model)

if __name__ == "__main__":
    main()
