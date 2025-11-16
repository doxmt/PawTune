import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "emotion_efficientnet_b0.pth"  
IMG_PATH = "test12.jpeg"  
CLASSES = ["angry", "happy", "sad"]  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open(IMG_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_label = CLASSES[pred_idx]

print(f"예측 감정: {pred_label}")
print(f"클래스별 확률:")
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<6}: {probs[i]*100:.2f}%")
