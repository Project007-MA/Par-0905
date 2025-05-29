# Imports and Setup
import os, cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import matplotlib.pyplot as plt
# Image Preprocessing
def apply_augmentations(image):
    transform = A.Compose([
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image=image)['image']
def load_images(paths, labels_map):
    images, labels = [], []
    for path, label in zip(paths, labels_map):
        if not os.path.exists(path):
            print(f":warning: Path not found: {path}")
            continue
        for fname in os.listdir(path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(path, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = apply_augmentations(img)
                    images.append(img.transpose(2, 0, 1))  # CxHxW
                    labels.append(label)
    return np.array(images), np.array(labels)
# Radiomics Features
def extract_radiomics(images):
    features = []
    for img in images:
        gray = np.mean(img, axis=0).astype(np.uint8)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        features.append(hist)
    return np.array(features)
# CNN + ViT Hybrid Feature Extractor
class CNN_ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()
        self.linear_proj = nn.Linear(768, 128)
    def forward(self, x):
        x_vit = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        vit_feat = self.linear_proj(self.vit(x_vit))
        cnn_feat = self.cnn(x).view(x.size(0), -1)
        return torch.cat([cnn_feat, vit_feat], dim=1)
def extract_features(model, images):
    model.eval().to(device)
    loader = DataLoader(TensorDataset(torch.tensor(images, dtype=torch.float32)), batch_size=8)
    features = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            features.append(model(batch).cpu().numpy())
    return np.concatenate(features)

# Dimensionality Reduction
def fuse_and_reduce(radiomics, cnn_vit, target_dim=256):
    fused = np.concatenate([radiomics, cnn_vit], axis=1)
    scaled = StandardScaler().fit_transform(fused)
    pca = PCA(n_components=min(target_dim, min(scaled.shape)))
    return pca.fit_transform(scaled)

# 1D CNN Classifier
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear((input_dim // 4) * 256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features[:, np.newaxis, :], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        BCE = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        return (self.alpha * (1 - pt) ** self.gamma * BCE).mean()
    
# === MAIN PIPELINE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = r"D:\Mahes\parkinson\mri"
img_paths = [os.path.join(data_path, 'normal'), os.path.join(data_path, 'parkinsons_dataset')]
labels_map = [0, 1]
images, labels = load_images(img_paths, labels_map)
if len(images) == 0:
    raise ValueError(":x: No images loaded. Check the paths and file formats.")
images_train, images_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)
radiomics_train = extract_radiomics(images_train)
radiomics_test = extract_radiomics(images_test)
model_feat = CNN_ViT()
cnn_vit_train = extract_features(model_feat, images_train)
cnn_vit_test = extract_features(model_feat, images_test)
fused_train = fuse_and_reduce(radiomics_train, cnn_vit_train)
fused_test = fuse_and_reduce(radiomics_test, cnn_vit_test)
fused_train, y_train = SMOTE().fit_resample(fused_train, y_train)
train_ds = FeatureDataset(fused_train, y_train)
test_ds = FeatureDataset(fused_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)
model = CNN1DClassifier(input_dim=fused_train.shape[1]).to(device)
loss_fn = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Training Loop
best_loss, patience, counter = float('inf'), 7, 0
for epoch in range(50):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x).squeeze()
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze()
            probs.extend(out.cpu().numpy())
            preds.extend((out > 0.5).float().cpu().numpy())
            labels.extend(y.cpu().numpy())
    acc = np.mean(np.array(preds) == np.array(labels))
    auc = roc_auc_score(labels, probs)
    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {acc*100:.2f}% | AUC: {auc:.4f}")
    if train_loss < best_loss:
        best_loss = train_loss
        counter = 0
        torch.save(model.state_dict(), "best_model_99.h5")
    else:
        counter += 1
        if counter >= patience:
            print(":black_square_for_stop: Early stopping.")
            break
        
# Final Evaluation
model.load_state_dict(torch.load("best_model_99.h5"))
model.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x).squeeze()
        probs = out.cpu().numpy()
        pred = (out > 0.5).float().cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(pred)
        all_labels.extend(y.cpu().numpy())
print(f"\n:white_check_mark: Final Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)) * 100:.2f}%")
print(f":chart_with_upwards_trend: AUC Score: {roc_auc_score(all_labels, all_probs):.4f}")
print(f":pushpin: F1: {f1_score(all_labels, all_preds):.2f} | Precision: {precision_score(all_labels, all_preds):.2f} | Recall: {recall_score(all_labels, all_preds):.2f}")
print(f"\n:bar_chart: Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
print("\n:clipboard: Classification Report:\n", classification_report(all_labels, all_preds, target_names=['Normal', "Parkinson's"]))