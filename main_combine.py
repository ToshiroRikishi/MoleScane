import os
from ultralytics import YOLO
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm

class CombinedSkinCancerSystem:
    def __init__(self, base_dir):
        """Инициализация с базовой директорией проекта."""
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / 'dataset'
        self.processed_dir = self.base_dir / 'processed_dataset'
        self.seg_processed_dir = self.base_dir / 'seg_processed_dataset'
        self.cls_processed_dir = self.base_dir / 'cls_processed_dataset'
        self.model_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        # Создаем необходимые директории
        for dir_path in [self.model_dir, self.results_dir, self.processed_dir,
                        self.seg_processed_dir, self.cls_processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
    def create_segmentation_yaml(self):
        """Создает YAML конфигурацию для датасета сегментации."""
        yaml_path = self.seg_processed_dir / 'dataset.yaml'
        
        data_yaml = {
            'path': str(self.seg_processed_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.class_names,
            'nc': len(self.class_names)
        }

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        return yaml_path

    def create_classification_yaml(self):
        """Создает YAML конфигурацию для датасета классификации."""
        yaml_path = self.cls_processed_dir / 'dataset.yaml'
        
        data_yaml = {
            'path': '.',
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        return yaml_path

    def prepare_folders(self):
        """Подготовка структуры папок для обоих типов данных."""
        splits = ['train', 'val', 'test']
        
        # Для сегментации
        for split in splits:
            (self.seg_processed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.seg_processed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Для классификации
        for split in splits:
            for cls in self.class_names:
                (self.cls_processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def convert_mask_to_yolo(self, mask, image_size):
        """Конвертация бинарной маски в формат YOLO."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        points = contour.reshape(-1, 2)
        points = points.astype(float)
        points[:, 0] /= image_size[1]
        points[:, 1] /= image_size[0]
        
        return points

    def process_image_for_classification(self, src_img, mask_path, dst_path):
        """Обработка изображения для классификации с использованием сегментации."""
        # Читаем изображение и маску
        img = cv2.imread(str(src_img))
        if img is None:
            return False
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False

        # Находим контуры в маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # Получаем наибольший контур
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)

        # Добавляем отступ
        padding = int(max(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        # Вырезаем область интереса
        roi = img[y1:y2, x1:x2]
        
        # Сохраняем обработанное изображение
        cv2.imwrite(str(dst_path), roi)
        return True

    def process_dataset(self):
        """Обработка и разделение датасета для обоих типов моделей."""
        print("Processing dataset...")

        # Чтение CSV с разметкой
        annot = pd.read_csv(self.dataset_dir / 'GroundTruth.csv')
        class_columns = annot.columns[1:]

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
        processed_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, split_data in splits_data.items():
            print(f"Processing {split_name} split...")
            
            for idx, row in tqdm(split_data.iterrows(), total=len(split_data)):
                img_id = row['image']
                class_values = [row[col] for col in class_columns]
                class_idx = class_values.index(1)
                class_name = self.class_names[class_idx]
                
                # Исходные пути
                src_img = self.dataset_dir / 'images' / f'{img_id}.jpg'
                mask_path = self.dataset_dir / 'masks' / f'{img_id}_segmentation.png'
                
                if not (src_img.exists() and mask_path.exists()):
                    continue

                # Для сегментации
                seg_img_path = self.seg_processed_dir / split_name / 'images' / f'{img_id}.jpg'
                seg_label_path = self.seg_processed_dir / split_name / 'labels' / f'{img_id}.txt'
                
                # Для классификации
                cls_img_path = self.cls_processed_dir / split_name / class_name / f'{img_id}.jpg'
                
                # Копируем и обрабатываем для сегментации
                shutil.copy(src_img, seg_img_path)
                
                # Создаем метку для сегментации
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(str(src_img))
                h, w = img.shape[:2]
                yolo_segments = self.convert_mask_to_yolo(mask, (h, w))
                
                if yolo_segments is not None:
                    with open(seg_label_path, 'w') as f:
                        points_flat = yolo_segments.reshape(-1)
                        points_str = ' '.join(map(str, points_flat))
                        f.write(f"{class_idx} {points_str}\n")
                
                # Обрабатываем для классификации
                if self.process_image_for_classification(src_img, mask_path, cls_img_path):
                    processed_counts[split_name] += 1

        return processed_counts

    def train_segmentation_model(self, yaml_path):
        """Обучение модели сегментации."""
        print("Training segmentation model...")
        
        device = "0" if torch.cuda.is_available() else "cpu"
        model = YOLO("yolo11n-seg.pt")

        results = model.train(
            data=str(yaml_path),
            epochs=10,
            imgsz=640,
            device=device,
            batch=16,
            patience=20,
            save=True,
            project=str(self.results_dir),
            name="skin_cancer_segmentation_3",
            exist_ok=True
        )

        metrics = model.val()
        export_path = self.model_dir / "segmentation_model.pt"
        shutil.copy(str(self.results_dir / "skin_cancer_segmentation_3" / "weights" / "best.pt"),
                   export_path)

        return model, metrics, export_path

    def train_classification_model(self, yaml_path):
        """Обучение модели классификации."""
        print("Training classification model...")
        
        device = "0" if torch.cuda.is_available() else "cpu"
        model = YOLO("yolo11n-cls.pt")

        results = model.train(
            data=str(yaml_path),
            epochs=10,
            imgsz=224,
            device=device,
            batch=32,
            patience=20,
            save=True,
            project=str(self.results_dir),
            name="skin_cancer_classification_3",
            exist_ok=True
        )

        metrics = model.val()
        export_path = self.model_dir / "classification_model.pt"
        shutil.copy(str(self.results_dir / "skin_cancer_classification_3" / "weights" / "best.pt"),
                   export_path)

        return model, metrics, export_path

    def create_combined_model(self):
        """Создание комбинированной модели."""
        seg_model_path = self.model_dir / "segmentation_model.pt"
        cls_model_path = self.model_dir / "classification_model.pt"
        
        if not (seg_model_path.exists() and cls_model_path.exists()):
            raise FileNotFoundError("Trained models not found")
            
        return CombinedModel(str(seg_model_path), str(cls_model_path))

    def run(self):
        """Основной метод запуска всего процесса."""
        try:
            print("Starting combined skin cancer detection project...")
            
            # Подготовка данных
            self.prepare_folders()
            dataset_stats = self.process_dataset()
            print("Dataset statistics:", dataset_stats)
            
            # Создание конфигураций
            seg_yaml = self.create_segmentation_yaml()
            cls_yaml = self.create_classification_yaml()
            
            # Обучение моделей
            seg_model, seg_metrics, seg_path = self.train_segmentation_model(seg_yaml)
            cls_model, cls_metrics, cls_path = self.train_classification_model("/home/user/MoleScane/seg_processed_dataset")
            
            # Создание комбинированной модели
            combined_model = self.create_combined_model()
            
            print("Project execution completed successfully!")
            return combined_model, {
                'segmentation': seg_metrics,
                'classification': cls_metrics
            }

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise

class CombinedModel:
    def __init__(self, seg_model_path, cls_model_path):
        """Инициализация комбинированной модели."""
        self.seg_model = YOLO(seg_model_path)
        self.cls_model = YOLO(cls_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    def preprocess_with_segmentation(self, image):
        """Предобработка изображения с использованием сегментации."""
        seg_results = self.seg_model.predict(
            image,
            conf=0.25,
            show=False,
            save=False
        )
        
        if len(seg_results) == 0 or len(seg_results[0].masks) == 0:
            return image, 0.0
            
        mask = seg_results[0].masks.data[0].cpu().numpy()
        segment_conf = float(seg_results[0].boxes.conf[0])
        
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image, 0.0
            
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        padding = int(max(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi = image[y1:y2, x1:x2]
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask[y1:y2, x1:x2].astype(np.uint8))
        
        return masked_roi, segment_conf
    
    def predict(self, image_path, conf_threshold=0.5):
        """Предсказание с использованием комбинированной модели."""
        # Загрузка изображения
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Предобработка с сегментацией
        processed_image, seg_conf = self.preprocess_with_segmentation(image)
        
        # Классификация
        cls_results = self.cls_model.predict(
            processed_image,
            conf=0.25,
            show=False,
            save=False
        )
        
        if len(cls_results) == 0:
            return {
                'class': None,
                'confidence': 0.0,
                'segmentation_confidence': seg_conf,
                'classification_confidence': 0.0,
                'is_reliable': False
            }
        
        probs = cls_results[0].probs
        predicted_class = int(probs.top1)
        cls_conf = float(probs.top1conf)
        
        # Вычисление комбинированной уверенности
        combined_conf = seg_conf * cls_conf
        
        # Формирование результата
        result = {
            'class': self.class_names[predicted_class],
            'confidence': float(combined_conf),
            'segmentation_confidence': float(seg_conf),
            'classification_confidence': float(cls_conf),
            'is_reliable': combined_conf >= conf_threshold,
            'all_probabilities': {
                self.class_names[i]: float(probs.data[i])
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def predict_batch(self, image_paths, conf_threshold=0.5):
        """Пакетное предсказание для нескольких изображений."""
        results = []
        for path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.predict(path, conf_threshold)
                results.append({
                    'image_path': str(path),
                    'prediction': result
                })
            except Exception as e:
                results.append({
                    'image_path': str(path),
                    'error': str(e)
                })
        return results
    
    def evaluate_on_dataset(self, dataset_path, ground_truth_csv):
        """Оценка производительности модели на тестовом наборе данных."""
        # Загрузка ground truth данных
        gt_data = pd.read_csv(ground_truth_csv)
        gt_data.set_index('image', inplace=True)
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        class_metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in self.class_names}
        
        # Проход по всем изображениям в датасете
        image_files = list(Path(dataset_path).glob('*.jpg'))
        for img_path in tqdm(image_files, desc="Evaluating model"):
            img_id = img_path.stem
            if img_id not in gt_data.index:
                continue
                
            # Получение ground truth класса
            true_class = None
            for cls in self.class_names:
                if gt_data.loc[img_id, cls] == 1:
                    true_class = cls
                    break
            
            if true_class is None:
                continue
                
            # Получение предсказания
            try:
                prediction = self.predict(img_path)
                if prediction['is_reliable']:
                    total_predictions += 1
                    predicted_class = prediction['class']
                    
                    # Обновление метрик
                    if predicted_class == true_class:
                        correct_predictions += 1
                        class_metrics[predicted_class]['tp'] += 1
                    else:
                        class_metrics[predicted_class]['fp'] += 1
                        class_metrics[true_class]['fn'] += 1
                    
                    results.append({
                        'image_id': img_id,
                        'true_class': true_class,
                        'predicted_class': predicted_class,
                        'confidence': prediction['confidence'],
                        'correct': predicted_class == true_class
                    })
            except Exception as e:
                print(f"Error processing {img_id}: {str(e)}")
        
        # Вычисление метрик
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        class_scores = {}
        for cls in self.class_names:
            tp = class_metrics[cls]['tp']
            fp = class_metrics[cls]['fp']
            fn = class_metrics[cls]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_scores[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # Формирование итогового отчета
        evaluation_report = {
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'class_metrics': class_scores,
            'individual_results': results
        }
        
        return evaluation_report

def main():
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    system = CombinedSkinCancerSystem(project_dir)
    
    try:
        # Обучение моделей и создание комбинированной системы
        combined_model, training_metrics = system.run()
        
        print("\nTraining metrics:")
        print("Segmentation metrics:")
        print(f"mAP: {training_metrics['segmentation'].seg.map}")
        print("\nClassification metrics:")
        print(f"Accuracy: {training_metrics['classification'].top1}")
        
        # Оценка на тестовом наборе
        test_dataset_path = system.dataset_dir / 'test'
        ground_truth_path = system.dataset_dir / 'GroundTruth.csv'
        
        if test_dataset_path.exists() and ground_truth_path.exists():
            print("\nEvaluating combined model on test dataset...")
            evaluation_results = combined_model.evaluate_on_dataset(
                test_dataset_path,
                ground_truth_path
            )
            
            print("\nEvaluation Results:")
            print(f"Overall Accuracy: {evaluation_results['accuracy']:.3f}")
            print("\nPer-class metrics:")
            for cls, metrics in evaluation_results['class_metrics'].items():
                print(f"\n{cls}:")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1-score: {metrics['f1_score']:.3f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()