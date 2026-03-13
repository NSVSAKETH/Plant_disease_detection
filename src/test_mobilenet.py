import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from utils import device, get_dataloaders

# Path to test image
IMAGE_PATH = r"C:\Users\saket\Downloads\plant_disease_project\plant_disease_project\c.jpg"

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Load class names
_, _, test_loader, num_classes = get_dataloaders()
class_names = test_loader.dataset.classes

# Helper: load weights safely
def load_weights(model, path):
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

# Load MobileNetV2 model
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(1280, num_classes)

mobilenet = load_weights(mobilenet, "models/mobilenet.pth")

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    outputs = mobilenet(input_tensor)
    probs = F.softmax(outputs, dim=1)
    confidence, pred = probs.max(dim=1)

# Output
pred_class = class_names[pred.item()]
conf_percent = confidence.item() * 100

print("\n🖼️ IMAGE TEST RESULT")
print("-" * 40)
print("Model Used  : MobileNetV2")
print("Prediction  :", pred_class)
print(f"Confidence  : {conf_percent:.2f}%")
print("-" * 40)