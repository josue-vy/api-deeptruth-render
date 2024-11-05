from fastapi import APIRouter, Form, File, UploadFile
from utils.dependencies import error_handling_dependency
from controllers import consultFaceController
router = APIRouter()


@router.post("/face/consult")
async def consultFace(imagen: UploadFile = File(...)):return await consultFaceController(imagen)