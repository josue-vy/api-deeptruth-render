# src/controllers/faceController.py
from fastapi import UploadFile, HTTPException
from utils import validParams
from services.faceService import deepFaceServiceInstance

async def consultFaceController(imagen: UploadFile):
    if(not validParams.checkParams(imagen)):
        raise HTTPException(status_code=400, detail="Invalid parameters")
    return await deepFaceServiceInstance.check_image_real_or_fake(imagen)
