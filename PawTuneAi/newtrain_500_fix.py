import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import random
import numpy as np

# ==============================
# 🧩 재현성
# ==============================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ==============================
# ⚙️ 설정
# ==============================
BATCH_SIZE = 8
EPOCHS = 30
# 핵심: 백본/헤드 서로 다른 LR
LR_BACKBONE = 2e-5
LR_CLASSIFIER = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 4  # angry, happy, relaxed, sad
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 📦 데이터셋 + (표정 보존형) 증강
# ==============================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # 과도한 크롭 완화
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.4, hue=0.06),
    transforms.RandomGrayscale(p=0.1),                     # 살짝만
    transforms.RandomAdjustSharpness(1.8, p=0.2),
    transforms.RandomAutocontrast(p=0.25),
    transforms.RandomRotation(15),
    # 원근/아핀 제거 → 표정 왜곡 최소화
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("emotion_dataset/train", transform=train_tf)
val_ds   = datasets.ImageFolder("emotion_dataset/val",   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================
# 🧠 모델
# ==============================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

# 전체 파인튜닝 (데이터 충분)
for p in model.features.parameters():
    p.requires_grad = True

model = model.to(DEVICE)

# ==============================
# ⚖️ 손실함수 (라벨 스무딩)
# ==============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# ==============================
# 🔧 Optimizer (파라미터 그룹)
# ==============================
backbone_params  = []
classifier_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith("classifier"):
        classifier_params.append(p)
    else:
        backbone_params.append(p)

optimizer = AdamW([
    {"params": backbone_params,   "lr": LR_BACKBONE},
    {"params": classifier_params, "lr": LR_CLASSIFIER},
], weight_decay=WEIGHT_DECAY)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==============================
# 💾 체크포인트
# ==============================
best_acc = 0.0
best_path = "emotion_newtrain_b0_4class_500_best.pth"

print(f"\n🚀 Training 시작 (Device: {DEVICE}) | Classes: {train_ds.classes}")
for epoch in range(1, EPOCHS+1):
    # ----- Train -----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # ----- Validation -----
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Acc: {val_acc*100:.2f}%")

    # 스케줄러 스텝
    scheduler.step()

    # 베스트 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print(f"  🔖 New best! Saved to {best_path} (Val Acc: {best_acc*100:.2f}%)")

# 마지막 전체 모델도 저장
final_path = "emotion_newtrain_b0_4class_500_final.pth"
torch.save(model.state_dict(), final_path)
print(f"\n✅ 학습 완료! Best: {best_acc*100:.2f}% | "
      f"Saved: {best_path} & {final_path}")
