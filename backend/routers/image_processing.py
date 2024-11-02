# # backend/routers/image_processing.py
# from fastapi import APIRouter

# router = APIRouter()

# @router.get("/")
# async def get_image_processing_info():
#     return {"message": "Image processing endpoint"}
# ____________________________________________________________________________________________________
# from fastapi import APIRouter, UploadFile, File
# from pathlib import Path
# from ultralytics import YOLO
# from PIL import Image
# import io

# router = APIRouter()

# # Путь к модели детекции
# model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"
# output_dir = "/home/user/MoleScane/images_show"
# model = YOLO(model_path)

# # Убедитесь, что директория для сохранения результатов существует
# output_path = Path(output_dir)
# output_path.mkdir(parents=True, exist_ok=True)

# @router.post("/detect")
# async def detect_image(file: UploadFile = File(...)):
#     # Загружаем изображение из файла
#     image = Image.open(io.BytesIO(await file.read()))

#     # Выполняем предсказание
#     results = model(image, conf=0.25)

#     # Сохраняем изображения с детекциями
#     for i, result in enumerate(results):
#         # Генерируем путь для сохранения детектированного изображения
#         detected_image_path = output_path / f"detected_image_{i}.jpg"
        
#         # Отрисовываем результаты на изображении
#         annotated_image = result.plot()
        
#         # Сохраняем отрисованное изображение
#         annotated_image.save(detected_image_path)
        
#     return {"message": "Detection complete", "output_path": str(detected_image_path)}

#_________________________________________________________________________________________

# #_________________________________________________________________________________________
# from fastapi import APIRouter, UploadFile, File, Response
# from pathlib import Path
# from ultralytics import YOLO
# from PIL import Image
# import io

# router = APIRouter()

# # Путь к модели детекции
# model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"
# output_dir = "/home/user/MoleScane/images_show"
# model = YOLO(model_path)

# # Убедитесь, что директория для сохранения результатов существует
# output_path = Path(output_dir)
# output_path.mkdir(parents=True, exist_ok=True)

# @router.post("/detect")
# async def detect_image(file: UploadFile = File(...)):
#     # Загружаем изображение из файла
#     image = Image.open(io.BytesIO(await file.read()))

#     # Выполняем предсказание
#     results = model(image, conf=0.25)

#     # Сохраняем и отображаем результат
#     for i, result in enumerate(results):
#         # Генерируем путь для сохранения детектированного изображения
#         detected_image_path = output_path / f"detected_image_{i}.jpg"
        
#         # Отрисовываем результаты на изображении и сохраняем его
#         annotated_image = Image.fromarray(result.plot()).convert("RGB")
#         annotated_image.save(detected_image_path)

#         # Конвертируем изображение в байты для отображения
#         img_byte_arr = io.BytesIO()
#         annotated_image.save(img_byte_arr, format='JPEG')
#         img_byte_arr = img_byte_arr.getvalue()

#     return Response(content=img_byte_arr, media_type="image/jpeg")
# # ____________________________________________________________________________

# from fastapi import APIRouter, UploadFile, File, Response
# from pathlib import Path
# from ultralytics import YOLO
# from PIL import Image
# import cv2
# import numpy as np
# import io

# router = APIRouter()

# # Путь к модели детекции
# model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"
# output_dir = "/home/user/MoleScane/images_show"
# model = YOLO(model_path)

# # Убедитесь, что директория для сохранения результатов существует
# output_path = Path(output_dir)
# output_path.mkdir(parents=True, exist_ok=True)

# @router.post("/detect")
# async def detect_image(file: UploadFile = File(...)):
#     # Загружаем изображение из файла и преобразуем его в BGR формат для OpenCV
#     image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     image_np = np.array(image)  # Конвертация в numpy
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Преобразование в BGR

#     # Выполняем предсказание
#     results = model(image_bgr, conf=0.25)

#     # Отрисовываем результаты на изображении OpenCV и сохраняем его
#     for result in results:
#         annotated_image = result.plot()  # Получаем аннотированное изображение

#     # Конвертируем обратно в RGB формат для сохранения в исходном цвете
#     annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#     annotated_image_pil = Image.fromarray(annotated_image_rgb)

#     # Сохраняем результат
#     detected_image_path = output_path / "detected_image.jpg"
#     annotated_image_pil.save(detected_image_path)

#     # Конвертируем изображение в байты для отображения
#     img_byte_arr = io.BytesIO()
#     annotated_image_pil.save(img_byte_arr, format='JPEG')
#     img_byte_arr = img_byte_arr.getvalue()

#     return Response(content=img_byte_arr, media_type="image/jpeg")

from fastapi import APIRouter, UploadFile, File, Response
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io

router = APIRouter()

# Путь к модели детекции
model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"
output_dir = "/home/user/MoleScane/images_show"
model = YOLO(model_path)

# Убедитесь, что директория для сохранения результатов существует
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    # Загружаем изображение из файла и преобразуем его в BGR формат для OpenCV
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_np = np.array(image)  # Конвертация в numpy
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Преобразование в BGR

    # Выполняем предсказание
    results = model(image_bgr, conf=0.25)

    # Отрисовываем результаты на изображении OpenCV и сохраняем его
    for result in results:
        annotated_image = result.plot()  # Получаем аннотированное изображение

    # Конвертируем обратно в RGB формат для сохранения в исходном цвете
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_image_pil = Image.fromarray(annotated_image_rgb)

    # Сохраняем результат
    detected_image_path = output_path / "detected_image.jpg"
    annotated_image_pil.save(detected_image_path)

    # Конвертируем изображение в байты для отображения
    img_byte_arr = io.BytesIO()
    annotated_image_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/jpeg")





