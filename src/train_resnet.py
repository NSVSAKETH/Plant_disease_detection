import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders, train_model, device

train_loader, val_loader, _, num_classes = get_dataloaders()

model = models.resnet50(pretrained=True)

for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
train_model(model, train_loader, val_loader, optimizer, "resnet")

# Fine-tuning
for name, p in model.named_parameters():
    if "layer4" in name:
        p.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
train_model(model, train_loader, val_loader, optimizer, "resnet", epochs=20)

# torch.save(model.state_dict(), "models/resnet.pth")
print("✅ ResNet training complete (checkpoints saved)")
