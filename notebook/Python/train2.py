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
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm  # Indikator Progress

# 1. Konfigurasi
CFG = {
    'seed': 42,
    'model_name': 'efficientnet_lite0', # Versi Lite lebih kencang
    'img_size': 224,
    'batch_size': 16, # Batch size dikecilkan agar tidak berat
    'epochs': 3,      # Cukup 3 epoch dulu untuk tes kecepatan
    'lr': 1e-4,
    'n_folds': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Matikan koneksi HF Hub agar tidak stuck di awal
os.environ['HF_HUB_OFFLINE'] = '1'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG['seed'])

# 2. Dataset Kelas
class FaceDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None, is_test=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, os.path.basename(img_path).split('.')[0]
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

# 3. Transformasi Sederhana (Agar Cepat)
train_transform = transforms.Compose([
    transforms.Resize((CFG['img_size'], CFG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = train_transform # Samakan agar cepat

# 4. Helper Functions
def get_train_data(base_path):
    file_paths, labels = [], []
    class_names = sorted(os.listdir(base_path))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    for cls in class_names:
        cls_folder = os.path.join(base_path, cls)
        if not os.path.isdir(cls_folder): continue
        for img in os.listdir(cls_folder):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(cls_folder, img))
                labels.append(class_to_idx[cls])
    return np.array(file_paths), np.array(labels), class_names

def get_test_data(csv_path, test_dir):
    df = pd.read_csv(csv_path)
    file_paths = []
    for img_id in df.iloc[:, 0]:
        img_id = str(img_id)
        path = os.path.join(test_dir, img_id)
        if not os.path.isfile(path):
            for ext in ['.jpg', '.jpeg', '.png']:
                if os.path.isfile(path + ext):
                    path += ext
                    break
        file_paths.append(path)
    return file_paths, df

# 5. Model
class FaceModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    def forward(self, x): return self.model(x)

# --- PROSES UTAMA ---

TRAIN_PATH = 'data/train'
TEST_DIR = 'data/test'
CSV_PATH = 'outputs/samplesubmission.csv'

if not os.path.exists(CSV_PATH):
    print(f"❌ Master, CSV tidak ada!")
else:
    train_paths, train_labels, class_names = get_train_data(TRAIN_PATH)
    test_paths, sample_df = get_test_data(CSV_PATH, TEST_DIR)
    num_classes = len(class_names)
    
    print(f"✅ Data Terdeteksi: Train={len(train_paths)}, Test={len(test_paths)}")

    skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
    best_model_path = 'best_model.pth'
    overall_best_f1 = 0

    for fold, (t_idx, v_idx) in enumerate(skf.split(train_paths, train_labels)):
        print(f"\n🚀 Fold {fold+1}/{CFG['n_folds']}")
        
        train_ds = FaceDataset(train_paths[t_idx], train_labels[t_idx], train_transform)
        val_ds = FaceDataset(train_paths[v_idx], train_labels[v_idx], val_transform)
        
        # Loader dioptimasi
        train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False)
        
        model = FaceModel(CFG['model_name'], num_classes).to(CFG['device'])
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
        criterion = nn.CrossEntropyLoss()
        
        fold_best_f1 = 0
        for epoch in range(CFG['epochs']):
            model.train()
            # Gunakan tqdm untuk melihat progres
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['epochs']}")
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(CFG['device']), lbls.to(CFG['device'])
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Eval
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    outputs = model(imgs.to(CFG['device']))
                    y_true.extend(lbls.numpy())
                    y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > fold_best_f1:
                fold_best_f1 = f1
                if f1 > overall_best_f1:
                    overall_best_f1 = f1
                    torch.save(model.state_dict(), best_model_path)
        
        print(f"✨ Fold {fold+1} Best F1: {fold_best_f1:.4f}")

    # C. Inference
    print("\n🎯 Membuat Submission...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loader = DataLoader(FaceDataset(test_paths, transform=val_transform, is_test=True), batch_size=CFG['batch_size'], shuffle=False)
    
    final_ids, final_preds = [], []
    for imgs, ids in tqdm(test_loader, desc="Predicting"):
        outputs = model(imgs.to(CFG['device']))
        final_ids.extend(ids)
        final_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    pd.DataFrame({'id': final_ids, 'label': final_preds}).to_csv('submission_final.csv', index=False)
    print("🔥 SELESAI MASTER! Cek file 'submission_final.csv'.")