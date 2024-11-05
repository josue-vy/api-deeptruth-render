import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import io

class DeepFaceService:
    instance = None

    def __init__(self):
        # Cargar el modelo `resnet18` desde el archivo local
        self.model = resnet18(pretrained=False)
        self.model.load_state_dict(torch.load('src/models/resnet18.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Etiquetas de salida del modelo
        self.labels = {0: "fake", 1: "real"}  # Ajuste según el modelo

        # Transformaciones de preprocesamiento de imágenes
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    async def check_image_real_or_fake(self, image_file):
        # Leer el contenido del archivo
        image_data = await image_file.read()
        # Abrir la imagen desde los datos leídos
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Preprocesar la imagen
        image = self.preprocess(image).unsqueeze(0)
        # Realizar la inferencia
        with torch.no_grad():
            outputs = self.model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        fake_probability = probabilities[0].item() * 100
        real_probability = probabilities[1].item() * 100
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
