import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

def train_model():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16 
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    IMG_SIZE = 224       
    DATA_DIR = '../Data/train_cropped/' 
    MODEL_SAVE_PATH = '../models/convnext_tiny_model.pth'

    os.makedirs('../models/', exist_ok=True)

    # Transform
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loader
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
    class_names = full_dataset.classes
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_dataset.dataset.transform = train_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=len(class_names), drop_rate=0.3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    print(f"🚀 Memulai Training ConvNeXt-Tiny di {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        
        # Eval
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
        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Model disimpan (Acc: {best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break

if __name__ == '__main__':
    train_model()