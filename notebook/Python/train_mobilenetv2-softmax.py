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
# 1. CONFIG
# ========================
CFG = {
    'seed': 42,
    'model_name': 'mobilenetv2_100',  # 🔥 MobileNetV2
    'img_size': 224,
    'batch_size': 32,
    'epochs': 5,
    'lr': 1e-4,
    'n_folds': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.environ['HF_HUB_OFFLINE'] = '1'

# ========================
# 2. SEED
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG['seed'])

print(f"🚀 Using device: {CFG['device']}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

# ========================
# 3. DATASET
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
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (CFG['img_size'], CFG['img_size']))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.original_ids[idx]

        return image, torch.tensor(self.labels[idx], dtype=torch.long)

# ========================
# 4. TRANSFORM
# ========================
transform_common = transforms.Compose([
    transforms.Resize((CFG['img_size'], CFG['img_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================
# 5. DATA HELPER
# ========================
def get_train_data(base_path):
    file_paths, labels = [], []
    class_names = sorted(os.listdir(base_path))

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    print("\n--- Mapping Kelas ---")
    for cls in class_names:
        cls_folder = os.path.join(base_path, cls)
        if not os.path.isdir(cls_folder):
            continue

        print(f"{class_to_idx[cls]} -> {cls}")

        for img in os.listdir(cls_folder):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(cls_folder, img))
                labels.append(class_to_idx[cls])

    return np.array(file_paths), np.array(labels), len(class_names), idx_to_class


def get_test_data(csv_path, test_dir):
    df = pd.read_csv(csv_path)
    file_paths, original_ids = [], []

    for img_id in df.iloc[:, 0]:
        img_id_str = str(img_id)
        original_ids.append(img_id_str)

        path = os.path.join(test_dir, img_id_str)
        if not os.path.isfile(path):
            for ext in ['.jpg', '.jpeg', '.png']:
                if os.path.isfile(path + ext):
                    path += ext
                    break

        file_paths.append(path)

    return file_paths, original_ids

# ========================
# 6. MODEL
# ========================
class FaceModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

# ========================
# MAIN
# ========================
TRAIN_PATH = 'data/train'
TEST_DIR = 'data/test'
CSV_PATH = 'outputs/samplesubmission.csv'

if not os.path.exists(CSV_PATH):
    print(f"❌ File {CSV_PATH} tidak ditemukan!")
else:
    train_paths, train_labels, num_classes, idx_to_class = get_train_data(TRAIN_PATH)
    test_paths, original_ids = get_test_data(CSV_PATH, TEST_DIR)

    print(f"\n✅ Model: {CFG['model_name']}")
    print(f"✅ Train: {len(train_paths)} | Test: {len(test_paths)}")

    skf = StratifiedKFold(
        n_splits=CFG['n_folds'],
        shuffle=True,
        random_state=CFG['seed']
    )

    best_model_path = 'best_mobilenetv2.pth'
    overall_best_f1 = 0

    for fold, (t_idx, v_idx) in enumerate(skf.split(train_paths, train_labels)):
        print(f"\n🚀 Fold {fold+1}/{CFG['n_folds']}")

        g = torch.Generator()
        g.manual_seed(CFG['seed'])

        train_loader = DataLoader(
            FaceDataset(train_paths[t_idx], train_labels[t_idx], transform_common),
            batch_size=CFG['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(CFG['seed'])
        )

        val_loader = DataLoader(
            FaceDataset(train_paths[v_idx], train_labels[v_idx], transform_common),
            batch_size=CFG['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        model = FaceModel(CFG['model_name'], num_classes).to(CFG['device'])
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
        criterion = nn.CrossEntropyLoss()

        fold_best_f1 = 0

        for epoch in range(CFG['epochs']):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for imgs, lbls in pbar:
                imgs = imgs.to(CFG['device'], non_blocking=True)
                lbls = lbls.to(CFG['device'], non_blocking=True)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # VALIDATION
            model.eval()
            y_true, y_pred = [], []

            with torch.no_grad():
                for imgs, lbls in val_loader:
                    outputs = model(imgs.to(CFG['device'], non_blocking=True))
                    y_true.extend(lbls.numpy())
                    y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            f1 = f1_score(y_true, y_pred, average='macro')

            if f1 > fold_best_f1:
                fold_best_f1 = f1
                if f1 > overall_best_f1:
                    overall_best_f1 = f1
                    torch.save(model.state_dict(), best_model_path)

        print(f"✨ Fold {fold+1} F1: {fold_best_f1:.4f}")

    # ========================
    # INFERENCE
    # ========================
    print("\n🎯 Generating Submission...")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_ds = FaceDataset(test_paths, transform=transform_common, is_test=True, original_ids=original_ids)

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    final_ids, final_preds_idx = [], []

    for imgs, ids in tqdm(test_loader, desc="Predicting"):
        outputs = model(imgs.to(CFG['device'], non_blocking=True))
        final_ids.extend(ids)
        final_preds_idx.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    final_preds_names = [idx_to_class[idx] for idx in final_preds_idx]

    pd.DataFrame({
        'id': final_ids,
        'label': final_preds_names
    }).to_csv('submission_mobilenetv2.csv', index=False)

    print("🔥 DONE! File: submission_mobilenetv2.csv")