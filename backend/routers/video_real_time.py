# backend/routers/image_processing.py
# from fastapi import APIRouter

# router = APIRouter()

# @router.get("/")
# async def get_image_processing_info():
#     return {"message": "video real time"}

# backend/routers/video_real_time.py
from fastapi import APIRouter, WebSocket
from ultralytics import YOLO
import cv2
import numpy as np

router = APIRouter()

# Инициализируем модель YOLO
model_path = "/home/user/MoleScane/combine_weights/best_detect.pt"
model = YOLO(model_path)

@router.websocket("/ws/video_real_time")
async def video_real_time_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Получаем кадр от клиента
            frame_data = await websocket.receive_bytes()
            np_frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            # Выполняем предсказание на текущем кадре
            results = model(frame, conf=0.25)
            annotated_frame = results[0].plot()

            # Кодируем и отправляем обработанный кадр обратно клиенту
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            await websocket.send_bytes(buffer.tobytes())
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        await websocket.close()
