import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==============================
# ⚙️ 설정
# ==============================
MODEL_PATH = "emotion_efficientnet_b0.pth"  # 학습된 모델 경로
IMG_PATH = "test5.jpeg"  # 테스트할 이미지 경로 (수정해서 사용)
CLASSES = ["angry", "happy", "sad"]  # 클래스 이름 (ImageFolder 순서와 같아야 함)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# 🧠 모델 로드
# ==============================
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==============================
# 🖼️ 이미지 전처리
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open(IMG_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ==============================
# 🔍 예측
# ==============================
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_label = CLASSES[pred_idx]

# ==============================
# 📊 결과 출력
# ==============================
print(f"예측 감정: {pred_label}")
print(f"클래스별 확률:")
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<6}: {probs[i]*100:.2f}%")
