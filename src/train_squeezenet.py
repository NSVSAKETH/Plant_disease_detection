import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import SqueezeNet1_1_Weights
from utils import get_dataloaders, train_model, device

# Load data
train_loader, val_loader, _, num_classes = get_dataloaders(batch_size=32)

# Load pretrained SqueezeNet (new API)
model = models.squeezenet1_1(
    weights=SqueezeNet1_1_Weights.DEFAULT
)

# Replace classifier
model.classifier[1] = nn.Conv2d(
    in_channels=512,
    out_channels=num_classes,
    kernel_size=1
)
model.num_classes = num_classes
model = model.to(device)

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,

)

# Train
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    model_name="squeezenet",
    epochs=40
)

# Rename checkpoint to final model at the end if desired, or just use the checkpoint
# torch.save(model.state_dict(), "models/squeezenet.pth")
print("✅ SqueezeNet training complete (checkpoints saved)")
