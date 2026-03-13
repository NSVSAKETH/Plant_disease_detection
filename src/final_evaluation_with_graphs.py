import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from utils import get_dataloaders, device

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
# Load FULL test data
# =====================================================
_, _, test_loader, num_classes = get_dataloaders(batch_size=32)
class_names = test_loader.dataset.classes
print(f"📊 Test samples: {len(test_loader.dataset)}")

# =====================================================
# Custom CNN (exact training architecture)
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
# Load models + file mapping
# =====================================================
models_dict = {}
model_files = {
    "ResNet": "models/resnet.pth",
    "MobileNet": "models/mobilenet.pth",
    "SqueezeNet": "models/squeezenet.pth",
    "CustomCNN": "models/customcnn.pth"
}

resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
models_dict["ResNet"] = load_weights(resnet, model_files["ResNet"])

mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(1280, num_classes)
models_dict["MobileNet"] = load_weights(mobilenet, model_files["MobileNet"])

squeezenet = models.squeezenet1_1(weights=None)
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, 1)
models_dict["SqueezeNet"] = load_weights(squeezenet, model_files["SqueezeNet"])

customcnn = CustomCNN(num_classes)
models_dict["CustomCNN"] = load_weights(customcnn, model_files["CustomCNN"])

# =====================================================
# Evaluation function
# =====================================================
def evaluate_model(model, model_key):
    y_true, y_pred, y_prob = [], [], []
    start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_prob.extend(probs)
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    end = time.time()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    roc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    size_mb = os.path.getsize(model_files[model_key]) / (1024 * 1024)
    inference_time = ((end - start) / len(y_true)) * 1000

    return acc, f1, roc, params, size_mb, inference_time, y_true_bin, np.array(y_prob)

# =====================================================
# Evaluate individual models
# =====================================================
results = []
roc_data = {}

print("\n📈 MODEL METRICS (FULL TEST)\n")

for name, model in models_dict.items():
    acc, f1, roc, params, size, time_ms, y_bin, y_prob = evaluate_model(model, name)
    results.append([name, acc*100, f1, roc, params, size, time_ms])
    roc_data[name] = (y_bin, y_prob)

    print(f"{name:10} | Acc: {acc*100:.2f}% | F1: {f1:.3f} | ROC-AUC: {roc:.3f} | "
          f"Params: {params:.2f}M | Size: {size:.1f}MB | {time_ms:.2f} ms/img")

# =====================================================
# Ensemble evaluation
# =====================================================
weights = [0.35, 0.30, 0.20, 0.15]
models_list = list(models_dict.values())

y_true, y_prob = [], []
start = time.time()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        probs = torch.zeros((x.size(0), num_classes), device=device)
        for m, w in zip(models_list, weights):
            probs += w * F.softmax(m(x), dim=1)
        y_prob.extend(probs.cpu().numpy())
        y_true.extend(y.numpy())

end = time.time()

y_pred = np.argmax(y_prob, axis=1)
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
roc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
inf_time = ((end - start) / len(y_true)) * 1000

results.append(["Ensemble", acc*100, f1, roc, "-", "-", inf_time])
roc_data["Ensemble"] = (y_true_bin, np.array(y_prob))

print(f"\nEnsemble   | Acc: {acc*100:.2f}% | F1: {f1:.3f} | ROC-AUC: {roc:.3f} | {inf_time:.2f} ms/img")

# =====================================================
# Save results table
# =====================================================
df = pd.DataFrame(results, columns=[
    "Model", "Accuracy (%)", "F1 Score", "ROC-AUC",
    "Parameters (M)", "Model Size (MB)", "Inference Time (ms/img)"
])

df.to_csv("models/final_metrics.csv", index=False)
print("\n✅ Metrics saved to models/final_metrics.csv")

# =====================================================
# Comparison graphs
# =====================================================
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Accuracy (%)"])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig("models/accuracy_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["F1 Score"])
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("models/f1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Inference Time (ms/img)"])
plt.title("Inference Time Comparison")
plt.ylabel("ms / image")
plt.tight_layout()
plt.savefig("models/inference_time_comparison.png", dpi=300)
plt.close()

# =====================================================
# ROC Curves
# =====================================================
plt.figure(figsize=(7,6))
for name, (y_bin, y_prob) in roc_data.items():
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (All Models)")
plt.legend()
plt.tight_layout()
plt.savefig("models/roc_curves.png", dpi=300)
plt.close()

print("\n🖼️ GRAPHS SAVED:")
print(" - models/accuracy_comparison.png")
print(" - models/f1_comparison.png")
print(" - models/inference_time_comparison.png")
print(" - models/roc_curves.png")
