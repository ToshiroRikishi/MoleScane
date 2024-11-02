# # backend/routers/image_processing.py
# from fastapi import APIRouter

# router = APIRouter()

# @router.get("/")
# async def get_image_processing_info():
#     return {"message": "Video processing endpoint"}


from fastapi import APIRouter, UploadFile, File, Response
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import io
import subprocess

router = APIRouter()

# Путь к модели детекции
model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"
output_dir = "/home/user/MoleScane/videos_show"
model = YOLO(model_path)

# Убедитесь, что директория для сохранения результатов существует
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

@router.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    # Сохраняем загруженное видео во временный файл
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(await file.read())
    temp_video.close()

    # Открываем видео с помощью OpenCV
    cap = cv2.VideoCapture(temp_video.name)
    if not cap.isOpened():
        return {"error": "Не удалось открыть видео"}

    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаем AVI видео с кодеком XVID
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    avi_path = output_path / "detected_video_suk.avi"
    out = cv2.VideoWriter(str(avi_path), fourcc, fps, (width, height))

    if not out.isOpened():
        return {"error": "Не удалось создать выходное видео"}

    # Обработка каждого кадра
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Выполняем предсказание и записываем кадр
        results = model(frame, conf=0.25)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    # Закрываем видеофайлы
    cap.release()
    out.release()

    # Преобразуем AVI в MP4 с помощью ffmpeg
    mp4_path = output_path / "detected_video.mp4"
    ffmpeg_command = f"ffmpeg -i {avi_path} -vcodec libx264 {mp4_path}"
    subprocess.run(ffmpeg_command, shell=True)

    # Возвращаем MP4 видео, если оно успешно создано
    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        with open(mp4_path, "rb") as f:
            video_bytes = f.read()
        return Response(content=video_bytes, media_type="video/mp4")
    else:
        return {"error": "Произошла ошибка при создании MP4 видео"}