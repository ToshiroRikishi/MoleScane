# backend/routers/image_processing.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_image_processing_info():
    return {"message": "Skin passport"}
