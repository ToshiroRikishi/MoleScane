# backend/main.py
from fastapi import FastAPI
from backend.routers import image_processing, video_processing, skin_passport
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import video_real_time

app = FastAPI()

# Настройка CORS для взаимодействия с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # разрешить доступ с любого источника
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение маршрутов
app.include_router(image_processing.router, prefix="/api/image")
app.include_router(video_processing.router, prefix="/api/video")
app.include_router(skin_passport.router, prefix="/api/skin")
app.include_router(video_real_time.router, prefix="/api/realTimeVideo")

@app.get("/")
async def root():
    return {"message": "Welcome to the main API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
