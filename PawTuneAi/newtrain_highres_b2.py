import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

BATCH_SIZE = 8
EPOCHS = 15                 
LR = 2e-5                   
NUM_CLASSES = 4
IMG_SIZE = 299              
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.4, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomAdjustSharpness(2, p=0.3),
    transforms.RandomAutocontrast(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("emotion_dataset/train", transform=train_tf)
val_ds   = datasets.ImageFolder("emotion_dataset/val",   transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("\n 모델 로드 중: EfficientNet-B2 (299×299 입력)")
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, NUM_CLASSES)
)

# 전체 fine-tuning
for p in model.features.parameters():
    p.requires_grad = True
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
print(f"\n🚀 Training 시작 (Device: {DEVICE}) | Classes: {train_ds.classes}")

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_correct, val_total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "emotion_highres_b2_best.pth")
        print(f"  🔖 New Best Model Saved (Val Acc: {best_acc*100:.2f}%)")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=train_ds.classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix (Best Val Acc: {best_acc*100:.2f}%)")
plt.tight_layout()
plt.savefig("cm_highres_b2.png")
plt.show()

torch.save(model.state_dict(), "emotion_highres_b2_final.pth")
print(f"\n 학습 완료! Best Val Acc: {best_acc*100:.2f}%")
