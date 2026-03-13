import torch
import torch.nn as nn
import time, os
import pandas as pd
from torchvision import models
from utils import get_dataloaders, device

# =====================================================
# Helper
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
print(f"📊 Test samples: {len(test_loader.dataset)}")

# =====================================================
# Custom CNN (EXACT MATCH)
# =====================================================
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
        return self.classifier(self.features(x))

# =====================================================
# Build models (HEAD FIRST!)
# =====================================================
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = load_weights(resnet, "models/resnet.pth")

mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(1280, num_classes)
mobilenet = load_weights(mobilenet, "models/mobilenet.pth")

squeezenet = models.squeezenet1_1(weights=None)
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, 1)
squeezenet = load_weights(squeezenet, "models/squeezenet.pth")

customcnn = CustomCNN(num_classes)
customcnn = load_weights(customcnn, "models/customcnn.pth")

models_dict = {
    "ResNet": resnet,
    "MobileNet": mobilenet,
    "SqueezeNet": squeezenet,
    "CustomCNN": customcnn
}

# =====================================================
# Metrics
# =====================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def evaluate(model):
    correct, total = 0, 0
    start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    end = time.time()
    acc = 100 * correct / total
    time_per_img = ((end - start) / total) * 1000
    return acc, time_per_img

# =====================================================
# Collect metrics
# =====================================================
rows = []

for name, model in models_dict.items():
    acc, inf_time = evaluate(model)
    params = count_params(model)
    size = model_size_mb(f"models/{name.lower()}.pth")

    rows.append([
        name,
        round(acc, 2),
        round(params, 2),
        round(size, 2),
        round(inf_time, 2)
    ])

df = pd.DataFrame(rows, columns=[
    "Model",
    "Accuracy (%)",
    "Parameters (M)",
    "Model Size (MB)",
    "Inference Time (ms/image)"
])

print("\n📊 PERFORMANCE METRICS\n")
print(df.to_string(index=False))

df.to_csv("models/performance_metrics.csv", index=False)
print("\n✅ Saved to models/performance_metrics.csv")
