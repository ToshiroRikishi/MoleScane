import os
from super_gradients.training import models

# Путь к чекпоинту
checkpoint_path = "/home/user/MoleScane/checkpoints/my_second_yolonas_run/RUN_20241029_093903_972029/ckpt_best.pth"

# Загрузка обученной модели
yolo_nas_l = models.get(
    "yolo_nas_l",
    num_classes=7,  # Количество классов, как в обучении
    checkpoint_path=checkpoint_path
)

# Установка пути для сохранения изображений
output_dir = "/home/user/MoleScane/images_show"
os.makedirs(output_dir, exist_ok=True)

# Выполнение предсказания и сохранение изображения
input_image_path = "/home/user/MoleScane/pred_images/ISIC_0024449.jpg"
output_path = os.path.join(output_dir, "detection_result.jpg")

# Выполнение предсказания для локального изображения
result = yolo_nas_l.predict(input_image_path, conf=0.25)
result.save(output_path)

print(f"Изображение с детекцией сохранено в {output_path}")
