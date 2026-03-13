import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from tqdm import tqdm

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loaders

def get_dataloaders(batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder("data/train", transform=train_transform)
    val_ds   = datasets.ImageFolder("data/val", transform=test_transform)
    test_ds  = datasets.ImageFolder("data/test", transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_ds.classes)

# Early Stopping

class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Training Function

def train_model(model, train_loader, val_loader, optimizer, model_name, epochs=40):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stop = EarlyStopping()

    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_checkpoint.pth"
    )

    best_model_path = os.path.join(
        checkpoint_dir, f"{model_name}.pth"
    )

    start_epoch = 0

    # ---------------- Resume Training ----------------
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        early_stop.best_loss = checkpoint['best_loss']

        print(f"▶ Resuming from epoch {start_epoch+1}")

    model.to(device)

    # ---------------- Training Loop ----------------
    for epoch in range(start_epoch, epochs):

        # -------- TRAIN --------
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs} [Train]")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct_train += (out.argmax(1) == y).sum().item()
            total_train += y.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct_train / total_train:.2f}%"
            })

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        pbar_val = tqdm(val_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for x, y in pbar_val:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss_v = criterion(out, y)

                val_loss += loss_v.item()
                correct_val += (out.argmax(1) == y).sum().item()
                total_val += y.size(0)

                pbar_val.set_postfix({
                    "loss": f"{loss_v.item():.4f}",
                    "acc": f"{100 * correct_val / total_val:.2f}%"
                })

        # -------- Metrics --------
        val_acc = 100 * correct_val / total_val
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"✨ Epoch {epoch+1} Summary | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # -------- Save Checkpoint --------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": early_stop.best_loss,
            "val_acc": val_acc
        }, checkpoint_path)

        # -------- Save Best Model --------
        if avg_val_loss <= early_stop.best_loss:
            torch.save(model.state_dict(), best_model_path)

        # -------- Early Stop --------
        if early_stop.step(avg_val_loss):
            print("⛔ Early stopping triggered")
            break
