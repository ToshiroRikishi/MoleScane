# from ultralytics import YOLO
# from pathlib import Path

# model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"  # замените на путь к вашей модели
# image_path = "/home/user/MoleScane/pred_images/photo_5260426290480144516_x.jpg"
# output_dir = "/home/user/MoleScane/images_show"
# output_path = Path(output_dir) / "detected_image.jpg"

# # Загружаем модель
# model = YOLO(model_path)
        
# # Выполняем предсказание
# results = model(image_path)
        
# # Сохраняем изображение с детекциями
# results.save(output_path)
# print(f"Detection result saved at: {output_path}")

from ultralytics import YOLO
from pathlib import Path

model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"  # замените на путь к вашей модели
image_path = "/home/user/MoleScane/pred_images/photo_5260426290480144516_x.jpg"
output_dir = "/home/user/MoleScane/images_show"

# Убедитесь, что директория для сохранения результатов существует
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Загружаем модель
model = YOLO(model_path)

# Выполняем предсказание
results = model(image_path)

# Сохраняем изображения с детекциями
for i, result in enumerate(results):
    detected_image_path = output_path / f"detected_image_{i}.jpg"  # Указываем расширение файла
    result.plot()
    result.save(str(detected_image_path))
    print(f"Detection result saved at: {detected_image_path}")

