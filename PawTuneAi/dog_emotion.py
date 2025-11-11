import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# -----------------------------
# 모델 설정
# -----------------------------
NUM_CLASSES = 4
IMG_SIZE = 299
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
CLASSES = ["angry", "happy", "relaxed", "sad"]

# -----------------------------
# 이미지 전처리
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------
# 모델 로드
# -----------------------------
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


# -----------------------------
# 예측 함수 (Flask에서 사용)
# -----------------------------
def predict_emotion(image):
    """이미지(PIL) 입력 → 예측 결과 문자열 반환"""
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    result = {
        "emotion": CLASSES[pred_idx],
        "confidence": round(probs[pred_idx].item() * 100, 2),
        "distribution": {CLASSES[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)}
    }
    return result
