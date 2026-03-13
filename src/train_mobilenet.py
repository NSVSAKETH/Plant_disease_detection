import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders, train_model, device

train_loader, val_loader, _, num_classes = get_dataloaders()

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, num_classes)

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
train_model(model, train_loader, val_loader, optimizer, "mobilenet")

# torch.save(model.state_dict(), "models/mobilenet.pth")
print("✅ MobileNet training complete (checkpoints saved)")
