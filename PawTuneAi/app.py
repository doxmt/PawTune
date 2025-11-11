from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # React 연동 허용

# -----------------------------
# 모델 설정
# -----------------------------
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

# -----------------------------
# 예측 API
# -----------------------------
@app.route("/analyze/dog", methods=["POST"])
def analyze_dog():
    if "image" not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # 감정별 확률 JSON으로 구성
    emotion_data = {CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(NUM_CLASSES)}
    pred_idx = probs.argmax()
    emotion = CLASSES[pred_idx]
    confidence = round(float(probs[pred_idx]) * 100, 2)

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "distribution": emotion_data
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
