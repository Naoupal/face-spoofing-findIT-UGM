import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from PIL import Image
import timm
from tqdm import tqdm

# --- CONFIGURATION ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 15
LR_CONV = 1e-4 [cite: 1]
LR_EFF = 1e-3
DATA_DIR = r'E:\Python - Project\face-spoofing-findIT-UGM\data\train'
TEST_DIR = r'E:\Python - Project\face-spoofing-findIT-UGM\data\test'
OUTPUT_DIR = r'E:\Python - Project\face-spoofing-findIT-UGM\outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TestDataset(Dataset):
    def __init__(self, root_dir, transform_c, transform_e):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform_c = transform_c
        self.transform_e = transform_e

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform_c(image), self.transform_e(image), self.image_files[idx].split('.')[0]

def train_single_model(model_name, img_size, lr, save_path, train_loader, val_loader, class_names):
    print(f"\n--- Training {model_name} (Size: {img_size}) ---")
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names), drop_rate=0.3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    patience, counter = 3, 0

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience: break
    
    return model

def main():
    # 1. Setup Data
    val_transform_c = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transform_e = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    class_names = full_dataset.classes
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_idx, val_idx = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # 2. Train ConvNeXt-Tiny
    train_loader_c = DataLoader(datasets.ImageFolder(DATA_DIR, transform=val_transform_c), batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(train_idx.indices))
    val_loader_c = DataLoader(datasets.ImageFolder(DATA_DIR, transform=val_transform_c), batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(val_idx.indices))
    model_c = train_single_model('convnext_tiny', 224, LR_CONV, os.path.join(OUTPUT_DIR, 'convnext_tiny.pth'), train_loader_c, val_loader_c, class_names)

    # 3. Train EfficientNet-B3
    train_loader_e = DataLoader(datasets.ImageFolder(DATA_DIR, transform=val_transform_e), batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(train_idx.indices))
    val_loader_e = DataLoader(datasets.ImageFolder(DATA_DIR, transform=val_transform_e), batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(val_idx.indices))
    model_e = train_single_model('efficientnet_b3', 300, LR_EFF, os.path.join(OUTPUT_DIR, 'efficientnet_b3.pth'), train_loader_e, val_loader_e, class_names)

    # 4. Ensemble Inference
    print("\n--- Generating Ensemble Submission ---")
    test_ds = TestDataset(TEST_DIR, val_transform_c, val_transform_e)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    results = []
    model_c.eval(); model_e.eval()
    with torch.no_grad():
        for in_c, in_e, ids in tqdm(test_loader, desc="Inference"):
            prob_c = torch.softmax(model_c(in_c.to(DEVICE)), dim=1)
            prob_e = torch.softmax(model_e(in_e.to(DEVICE)), dim=1)
            final_preds = torch.max((prob_c + prob_e) / 2, 1)[1]
            for i in range(len(ids)):
                results.append({'id': ids[i], 'label': class_names[final_preds[i].item()]})

    pd.DataFrame(results).to_csv('submission.csv', index=False)
    print("✅ Done, Master. File 'submission.csv' is ready.")

if __name__ == '__main__':
    main()