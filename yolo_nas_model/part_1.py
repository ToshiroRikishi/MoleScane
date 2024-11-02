# from super_gradients.training import models

# yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

# url = "https://previews.123rf.com/images/freeograph/freeograph2011/freeograph201100150/158301822-group-of-friends-gathering-around-table-at-home.jpg"
# yolo_nas_l.predict(url, conf=0.25).show()

import os
from super_gradients.training import models

# Загрузка модели
yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

# Установка пути для сохранения изображений
output_dir = "/home/user/MoleScane/images_show"
os.makedirs(output_dir, exist_ok=True)

# Выполнение предсказания и сохранение изображения
url = "https://previews.123rf.com/images/freeograph/freeograph2011/freeograph201100150/158301822-group-of-friends-gathering-around-table-at-home.jpg"
output_path = os.path.join(output_dir, "detection_result.jpg")
result = yolo_nas_l.predict(url, conf=0.25)
result.save(output_path)

print(f"Изображение с детекцией сохранено в {output_path}")

