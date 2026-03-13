"""Test ResNet model on a single image."""
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get class names from training data
train_dir = "data/train"
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("models/resnet.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test image - pick one from test set
test_class = "Tomato_Late_blight"
test_dir = f"data/test/{test_class}"
test_images = os.listdir(test_dir)
test_img_path = os.path.join(test_dir, test_images[0])

print(f"\nTest Image: {test_img_path}")
print(f"True Class: {test_class}")

# Load and preprocess image
img = Image.open(test_img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item() * 100

print(f"\nPrediction: {class_names[pred_idx]}")
print(f"Confidence: {confidence:.2f}%")

# Show top 5 predictions
print("\nTop 5 Predictions:")
top5_probs, top5_idx = probs.topk(5)
for i, (prob, idx) in enumerate(zip(top5_probs, top5_idx)):
    marker = " <-- TRUE" if class_names[idx] == test_class else ""
    print(f"  {i+1}. {class_names[idx]}: {prob.item()*100:.2f}%{marker}")

# Check if correct
if class_names[pred_idx] == test_class:
    print("\n[CORRECT] Model prediction is correct!")
else:
    print(f"\n[WRONG] Expected: {test_class}, Got: {class_names[pred_idx]}")
