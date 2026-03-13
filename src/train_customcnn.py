import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders, train_model, device

# -----------------------------
# Custom CNN
# -----------------------------
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


# -----------------------------
# Load Data
# -----------------------------
train_loader, val_loader, _, num_classes = get_dataloaders()

# -----------------------------
# Model
# -----------------------------
model = CustomCNN(num_classes).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# -----------------------------
# Train
# -----------------------------
train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    model_name="customcnn",
    epochs=40
)

print("✅ Custom CNN training complete")
