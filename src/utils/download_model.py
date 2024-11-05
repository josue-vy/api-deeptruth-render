import torch
from torchvision.models import resnet18

# Cargar el modelo `resnet18` preentrenado
model = resnet18(pretrained=True)
model.eval()

# Guardar el estado del modelo en un archivo local llamado `resnet18.pth`
torch.save(model.state_dict(), 'resnet18.pth')
print("Modelo guardado como 'resnet18.pth'")
