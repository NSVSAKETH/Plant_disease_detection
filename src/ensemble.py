import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import get_dataloaders, device

# =================================================
# Helper function to load weights safely
# =================================================
def load_weights(model, path):
    checkpoint = torch.load(path, map_location=device)

    # If checkpoint dict, extract model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

# Load test data
_, _, test_loader, num_classes = get_dataloaders(batch_size=32)

print("🚀 Starting Ensemble Evaluation...")
print(f"Number of classes: {num_classes}")


# ResNet50
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = load_weights(resnet, "models/resnet.pth")

# MobileNetV2
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(1280, num_classes)
mobilenet = load_weights(mobilenet, "models/mobilenet.pth")

# SqueezeNet
squeezenet = models.squeezenet1_1(weights=None)
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
squeezenet = load_weights(squeezenet, "models/squeezenet.pth")

# Custom CNN  (⚠️ MUST MATCH TRAINING EXACTLY)
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


customcnn = CustomCNN(num_classes)
customcnn = load_weights(customcnn, "models/customcnn.pth")

# Ensemble setup
models_list = [resnet, mobilenet, squeezenet, customcnn]
weights = [0.35, 0.30, 0.20, 0.15]  # must sum to 1.0

assert abs(sum(weights) - 1.0) < 1e-6, "Ensemble weights must sum to 1"

# Ensemble Evaluation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        probs = torch.zeros((images.size(0), num_classes), device=device)

        for model, w in zip(models_list, weights):
            probs += w * F.softmax(model(images), dim=1)

        preds = probs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\n✅ Ensemble Accuracy: {accuracy:.2f}%")
