import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import time

# =========================
# Class names (23 classes)
# =========================

class_names = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___Northern_Leaf_Blight",
"Corn_(maize)___healthy",
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Tomato_Bacterial_spot",
"Tomato_Early_blight",
"Tomato_Late_blight",
"Tomato_Leaf_Mold",
"Tomato_Septoria_leaf_spot",
"Tomato_Spider_mites_Two_spotted_spider_mite",
"Tomato__Target_Spot",
"Tomato__Tomato_YellowLeaf__Curl_Virus",
"Tomato__Tomato_mosaic_virus",
"Tomato_healthy"
]

# =========================
# Load ONNX model
# =========================

session = ort.InferenceSession(
    "mobilenet.onnx",
    providers=["CPUExecutionProvider"]
)

# =========================
# Image preprocessing
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =========================
# Get image path
# =========================

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "c.jpg"

# =========================
# Load image
# =========================

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Convert to numpy
input_numpy = input_tensor.numpy().astype(np.float32)

# =========================
# Run inference
# =========================

start = time.time()

outputs = session.run(None, {"input": input_numpy})

end = time.time()

# =========================
# Prediction
# =========================

probs = torch.softmax(torch.tensor(outputs[0]), dim=1)

confidence, pred = torch.max(probs,1)

prediction = class_names[pred.item()]
confidence = confidence.item()*100

# =========================
# Output
# =========================

print("\n🖼️ IMAGE TEST RESULT (ONNX)")
print("-"*40)
print("Model Used  : MobileNetV2 (ONNX)")
print("Prediction  :", prediction)
print(f"Confidence  : {confidence:.2f}%")
print(f"Inference   : {(end-start)*1000:.2f} ms")
print("-"*40)