import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from utils import get_dataloaders, device

# =====================================================
# Helper: safe weight loading
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
# Load test data
# =====================================================
_, _, test_loader, num_classes = get_dataloaders(batch_size=32)

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
# Load models
# =====================================================
models_list = []

resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
models_list.append(load_weights(resnet, "models/resnet.pth"))

mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(1280, num_classes)
models_list.append(load_weights(mobilenet, "models/mobilenet.pth"))

squeezenet = models.squeezenet1_1(weights=None)
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, 1)
models_list.append(load_weights(squeezenet, "models/squeezenet.pth"))

customcnn = CustomCNN(num_classes)
models_list.append(load_weights(customcnn, "models/customcnn.pth"))

# =====================================================
# Confidence-aware ensemble evaluation
# =====================================================
def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        all_probs = []
        all_weights = []

        for model in models_list:
            probs = F.softmax(model(images), dim=1)
            conf = probs.max(dim=1).values
            ent = entropy(probs)

            weight = conf / (ent + 1e-6)
            all_probs.append(probs)
            all_weights.append(weight)

        # Stack → shape [num_models, batch, classes]
        probs_stack = torch.stack(all_probs)
        weights_stack = torch.stack(all_weights)

        # Normalize weights per sample
        weights_norm = weights_stack / weights_stack.sum(dim=0, keepdim=True)

        # Weighted sum
        final_probs = torch.sum(
            weights_norm.unsqueeze(-1) * probs_stack,
            dim=0
        )

        preds = final_probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\n🧠 Smart Confidence-Aware Ensemble Accuracy: {accuracy:.2f}%")
