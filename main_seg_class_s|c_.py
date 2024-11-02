import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

class CombinedSkinCancerModel:
    def __init__(self, base_dir):
        """Инициализация комбинированной модели для сегментации и классификации."""
        self.base_dir = Path(base_dir)
        self.val_dir = self.base_dir / 'val' / 'images'
        self.model_seg = YOLO("/home/user/MoleScane/combine_weights/best_seg.pt")
        self.model_cls = YOLO("/home/user/MoleScane/combine_weights/best_cl.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Загружаем метки классов из файла GroundTruth.csv
        ground_truth_path = self.base_dir.parent / 'dataset' / 'GroundTruth.csv'
        self.ground_truth = pd.read_csv(ground_truth_path, index_col='image')

    def get_true_index(self, image_name):
        """Получает индекс класса из метки для данного изображения."""
        true_label_row = self.ground_truth.loc[image_name]
        true_idx = true_label_row[true_label_row == 1.0].index[0]  # Получаем индекс класса
        return list(self.ground_truth.columns).index(true_idx)  # Преобразуем в числовой индекс

    def validate(self):
        """Валидация моделей на валидационном датасете."""
        top1_correct = 0
        top5_correct = 0
        total_loss = 0
        total_samples = 0

        for img_path in tqdm(self.val_dir.glob("*.jpg"), desc="Validating"):
            image_name = img_path.stem  # Получаем имя изображения без расширения

            # Сегментация
            seg_result = self.model_seg.predict(source=str(img_path), device=self.device, verbose=False)
            
            if seg_result[0].masks is not None:
                mask = seg_result[0].masks.data[0].cpu().numpy()  # используем data
                mask = (mask * 255).astype(np.uint8)  # преобразуем к uint8
                
                # Проверяем размерность маски
                image = cv2.imread(str(img_path))
                if mask.shape[:2] != image.shape[:2]:  # изменяем размер, если не совпадает
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

                # Применяем маску к изображению
                mask_applied = cv2.bitwise_and(image, image, mask=mask)
                crop_img = cv2.resize(mask_applied, (224, 224))  # подгонка под классификатор
            else:
                image = cv2.imread(str(img_path))
                crop_img = cv2.resize(image, (224, 224))  # обрабатываем изображение без маски

            # Классификация
            cls_result = self.model_cls.predict(source=crop_img, device=self.device, verbose=False)
            preds = cls_result[0].probs.cpu().numpy()

            # Расчёт метрик
            true_idx = self.get_true_index(image_name)
            loss = -np.log(preds[true_idx])
            total_loss += loss

            # Точность Top-1 и Top-5
            top5_pred = np.argsort(preds)[-5:]
            if true_idx == top5_pred[-1]:
                top1_correct += 1
            if true_idx in top5_pred:
                top5_correct += 1

            total_samples += 1

        avg_loss = total_loss / total_samples
        top1_acc = top1_correct / total_samples
        top5_acc = top5_correct / total_samples

        return {
            "loss": avg_loss,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc
        }

def main():
    base_dir = "/home/user/MoleScane/processed_dataset"
    combined_model = CombinedSkinCancerModel(base_dir)
    metrics = combined_model.validate()
    
    print("\nValidation Metrics:")
    print(f"Loss: {metrics['loss']}")
    print(f"Top-1 Accuracy: {metrics['top1_acc']}")
    print(f"Top-5 Accuracy: {metrics['top5_acc']}")

if __name__ == "__main__":
    main()
