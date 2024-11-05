import torch
from torchvision import transforms
from PIL import Image
import os
import io

class DeepFaceService:
    instance = None

    def __init__(self, model_url="https://github.com/pytorch/vision/tree/main/references/classification"):
        """
        Inicializa la clase y carga el modelo preentrenado de PyTorch Hub.
        
        :param model_url: URL del modelo en PyTorch Hub.
        """
        # Configura la caché de PyTorch para almacenar el modelo en un directorio específico
        os.environ['TORCH_HOME'] = './src/torch_cache'

        # Cargar el modelo preentrenado de PyTorch Hub
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.model.eval()
        self.labels = {0: "fake", 1: "real"}  # Ajuste según el modelo

        # Transformaciones de preprocesamiento de imágenes
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    async def check_image_real_or_fake(self, image_file):
        """
        Consulta el modelo para determinar si una imagen es real o falsa.
        
        :param image_file: Archivo de imagen a consultar (de tipo UploadFile).
        :return: Diccionario con la etiqueta predicha por el modelo ("real" o "fake") y el porcentaje de probabilidad.
        """
        # Leer el contenido del archivo
        image_data = await image_file.read()

        # Abrir la imagen desde los datos leídos
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocesar la imagen
        image = self.preprocess(image).unsqueeze(0)

        # Realizar la inferencia
        with torch.no_grad():
            outputs = self.model(image)

        # Obtener las predicciones
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Para este ejemplo, el índice 0 es "fake" y el índice 1 es "real"
        fake_probability = probabilities[0].item() * 100
        real_probability = probabilities[1].item() * 100

        # Determinar la etiqueta basada en la probabilidad más alta
        predicted_label = "fake" if fake_probability > real_probability else "real"
        predicted_probability = max(fake_probability, real_probability)

        return {"label": predicted_label, "probability": predicted_probability}

    @staticmethod
    def getInstance():
        if DeepFaceService.instance is None:
            DeepFaceService.instance = DeepFaceService()
        return DeepFaceService.instance

# Instanciar el servicio
deepFaceServiceInstance = DeepFaceService.getInstance()
