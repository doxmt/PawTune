import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

NUM_CLASSES = 4
IMG_SIZE = 299
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["angry", "happy", "relaxed", "sad"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

model = models.efficientnet_b2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load("emotion_highres_b2_best.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    print("예측 감정:", CLASSES[pred_idx])
    print("확률 분포:")
    for i, p in enumerate(probs):
        print(f"   {CLASSES[i]}: {p*100:.2f}%")

if __name__ == "__main__":
    test_path = "test12.jpeg"  
    predict_image(test_path)
