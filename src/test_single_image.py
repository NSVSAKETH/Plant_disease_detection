import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from utils import device, get_dataloaders

# =====================================================
# CHANGE THIS: path to your test image
# =====================================================
IMAGE_PATH = r"C:\Users\saket\Downloads\plant_disease_project\plant_disease_project\bl.png"   # put your image path here
USE_ENSEMBLE = True       # True → ensemble, False → single model
MODEL_NAME = "resnet"     # used only if USE_ENSEMBLE = False
# options: resnet, mobilenet, squeezenet, customcnn

# =====================================================
# Image preprocessing (same as test transform)
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# Load class names
# =====================================================
_, _, test_loader, num_classes = get_dataloaders()
class_names = test_loader.dataset.classes

# =====================================================
# Helper: load weights safely
# =====================================================
def load_weights(model, path):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# =====================================================
# Custom CNN (same as training)
# =====================================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# =====================================================
# Load image
# =====================================================
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# =====================================================
# Load models
# =====================================================
models_dict = {}

resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
models_dict["resnet"] = load_weights(resnet, "models/resnet.pth")

mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(1280, num_classes)
models_dict["mobilenet"] = load_weights(mobilenet, "models/mobilenet.pth")

squeezenet = models.squeezenet1_1(weights=None)
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, 1)
models_dict["squeezenet"] = load_weights(squeezenet, "models/squeezenet.pth")

customcnn = CustomCNN(num_classes)
models_dict["customcnn"] = load_weights(customcnn, "models/customcnn.pth")

# =====================================================
# Prediction
# =====================================================
with torch.no_grad():

    if USE_ENSEMBLE:
        weights = [0.35, 0.30, 0.20, 0.15]
        probs = torch.zeros((1, num_classes), device=device)

        for model, w in zip(models_dict.values(), weights):
            probs += w * F.softmax(model(input_tensor), dim=1)

        confidence, pred = probs.max(dim=1)
        model_used = "Ensemble"

    else:
        model = models_dict[MODEL_NAME.lower()]
        probs = F.softmax(model(input_tensor), dim=1)
        confidence, pred = probs.max(dim=1)
        model_used = MODEL_NAME.capitalize()

# =====================================================
# Output
# =====================================================
pred_class = class_names[pred.item()]
conf_percent = confidence.item() * 100

print("\n🖼️ IMAGE TEST RESULT")
print("-" * 40)
print(f"Model Used   : {model_used}")
print(f"Prediction  : {pred_class}")
print(f"Confidence  : {conf_percent:.2f}%")
print("-" * 40)
