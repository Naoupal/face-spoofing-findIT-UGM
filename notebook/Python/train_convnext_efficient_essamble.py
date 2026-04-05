import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
import numpy as np
import random
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

# ========================
# CONFIG
# ========================
CFG = {
    'seed': 42,
    'models': ['efficientnet_b0', 'convnext_tiny'],  # 🔥 2 model
    'img_size': 224,
    'batch_size': 32,
    'epochs': 5,
    'lr': 1e-4,
    'n_folds': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# ========================
# SEED
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG['seed'])

print(f"🚀 Device: {CFG['device']}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

# ========================
# DATASET
# ========================
class FaceDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None, is_test=False, original_ids=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        self.original_ids = original_ids

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (CFG['img_size'], CFG['img_size']))

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, self.original_ids[idx]

        return img, torch.tensor(self.labels[idx], dtype=torch.long)

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((CFG['img_size'], CFG['img_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================
# DATA LOADER
# ========================
def get_train_data(base_path):
    file_paths, labels = [], []
    classes = sorted(os.listdir(base_path))

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    print("\nMapping kelas:")
    for cls in classes:
        print(f"{class_to_idx[cls]} -> {cls}")
        folder = os.path.join(base_path, cls)
        for img in os.listdir(folder):
            if img.endswith(('.jpg','.png','.jpeg')):
                file_paths.append(os.path.join(folder, img))
                labels.append(class_to_idx[cls])

    return np.array(file_paths), np.array(labels), len(classes), idx_to_class


def get_test_data(csv_path, test_dir):
    df = pd.read_csv(csv_path)
    paths, ids = [], []

    for img_id in df.iloc[:,0]:
        img_id = str(img_id)
        ids.append(img_id)

        path = os.path.join(test_dir, img_id)
        if not os.path.exists(path):
            for ext in ['.jpg','.png','.jpeg']:
                if os.path.exists(path+ext):
                    path += ext
                    break
        paths.append(path)

    return paths, ids

# ========================
# MODEL BUILDER
# ========================
class FaceModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)

        if "efficientnet" in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

        elif "convnext" in model_name:
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes)

        else:
            raise ValueError("Model tidak didukung")

    def forward(self, x):
        return self.model(x)

# ========================
# TRAIN FUNCTION
# ========================
def train_model(model_name, train_paths, train_labels):
    print(f"\n🔥 Training {model_name}")

    skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
    best_path = f"best_{model_name}.pth"
    best_score = 0

    for fold, (t_idx, v_idx) in enumerate(skf.split(train_paths, train_labels)):
        print(f"\nFold {fold+1}")

        train_loader = DataLoader(
            FaceDataset(train_paths[t_idx], train_labels[t_idx], transform),
            batch_size=CFG['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            FaceDataset(train_paths[v_idx], train_labels[v_idx], transform),
            batch_size=CFG['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        model = FaceModel(model_name, num_classes).to(CFG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(CFG['epochs']):
            model.train()
            for imgs, lbls in tqdm(train_loader):
                imgs = imgs.to(CFG['device'])
                lbls = lbls.to(CFG['device'])

                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()

            # VALIDATION
            model.eval()
            y_true, y_pred = [], []

            with torch.no_grad():
                for imgs, lbls in val_loader:
                    outputs = model(imgs.to(CFG['device']))
                    y_true.extend(lbls.numpy())
                    y_pred.extend(torch.argmax(outputs,1).cpu().numpy())

            f1 = f1_score(y_true, y_pred, average='macro')
            print(f"Epoch {epoch+1} F1: {f1:.4f}")

            if f1 > best_score:
                best_score = f1
                torch.save(model.state_dict(), best_path)

    return best_path

# ========================
# INFERENCE
# ========================
def predict(model_paths, test_paths, ids):
    test_loader = DataLoader(
        FaceDataset(test_paths, transform=transform, is_test=True, original_ids=ids),
        batch_size=CFG['batch_size'],
        shuffle=False
    )

    final_probs = None

    for model_name, path in model_paths:
        print(f"🔮 Predicting with {model_name}")

        model = FaceModel(model_name, num_classes).to(CFG['device'])
        model.load_state_dict(torch.load(path))
        model.eval()

        probs_list = []

        with torch.no_grad():
            for imgs, _ in test_loader:
                outputs = model(imgs.to(CFG['device']))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_list.append(probs)

        probs_all = np.concatenate(probs_list)

        if final_probs is None:
            final_probs = probs_all
        else:
            final_probs += probs_all  # 🔥 ensemble

    final_preds = np.argmax(final_probs, axis=1)
    return final_preds

# ========================
# MAIN
# ========================
TRAIN_PATH = 'data/train'
TEST_DIR = 'data/test'
CSV_PATH = 'outputs/samplesubmission.csv'

train_paths, train_labels, num_classes, idx_to_class = get_train_data(TRAIN_PATH)
test_paths, ids = get_test_data(CSV_PATH, TEST_DIR)

model_paths = []

# TRAIN SEMUA MODEL
for model_name in CFG['models']:
    path = train_model(model_name, train_paths, train_labels)
    model_paths.append((model_name, path))

# ENSEMBLE PREDICTION
preds = predict(model_paths, test_paths, ids)

pred_labels = [idx_to_class[i] for i in preds]

pd.DataFrame({
    'id': ids,
    'label': pred_labels
}).to_csv('submission_convnext_efficient.csv', index=False)

print("🔥 DONE! submission_convnext_efficient.csv siap!")