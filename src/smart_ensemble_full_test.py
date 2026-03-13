import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import get_dataloaders, device
import numpy as np

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
# Load FULL test dataset
# =====================================================
_, _, test_loader, num_classes = get_dataloaders(batch_size=32)
print(f"📊 Total test samples: {len(test_loader.dataset)}")

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
# Load all models
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

print("✅ All models loaded successfully")

# =====================================================
# Entropy (uncertainty) function
# =====================================================
def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

# =====================================================
# SMART ENSEMBLE FULL TEST
# =====================================================
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        all_probs = []
        all_weights = []

        # Run ALL models
        for model in models_list:
            probs = F.softmax(model(images), dim=1)
            confidence = probs.max(dim=1).values
            uncertainty = entropy(probs)

            # Confidence-aware weight
            weight = confidence / (uncertainty + 1e-6)

            all_probs.append(probs)
            all_weights.append(weight)

        # Stack results
        probs_stack = torch.stack(all_probs)        # [M, B, C]
        weights_stack = torch.stack(all_weights)    # [M, B]

        # Normalize weights per image
        weights_norm = weights_stack / weights_stack.sum(dim=0, keepdim=True)

        # Weighted ensemble
        final_probs = torch.sum(
            weights_norm.unsqueeze(-1) * probs_stack,
            dim=0
        )

        preds = final_probs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

# =====================================================
# Final accuracy
# =====================================================
accuracy = 100 * correct / total

print("\n🧠 SMART ENSEMBLE – FULL DATASET RESULT")
print(f"✅ Total Images Tested : {total}")
print(f"🎯 Ensemble Accuracy   : {accuracy:.2f}%")
