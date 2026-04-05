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

# 1. Konfigurasi Dasar & Random Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to: {seed}")

CFG = {
    'seed': 42,
    'model_name': 'efficientnet_b0',
    'img_size': 224,
    'batch_size': 32,
    'num_classes': 2, # Sesuaikan dengan jumlah kelas Master (misal: real vs fake)
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

set_seed(CFG['seed'])

# 2. Custom Dataset - Diperbaiki agar mencari ekstensi file dengan benar
class FaceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Ambil ID dari kolom pertama
        img_id = str(self.df.iloc[idx, 0])
        
        # Cari file dengan mencoba berbagai ekstensi jika ID tidak punya ekstensi
        img_path = os.path.join(self.img_dir, img_id)
        
        if not os.path.isfile(img_path):
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                if os.path.isfile(img_path + ext):
                    img_path = img_path + ext
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Master, gambar {img_id} tidak ditemukan di {self.img_dir}")

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_id

# 3. Transformasi
transform_test = transforms.Compose([
    transforms.Resize((CFG['img_size'], CFG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Model EfficientNet
class FaceModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(FaceModel, self).__init__()
        # Menggunakan parameter 'pretrained=True' (timm terbaru menggunakan 'pretrained=True')
        self.model = timm.create_model(model_name, pretrained=True)
        # Menyesuaikan classifier untuk EfficientNet
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 5. Pipeline Prediksi
def predict(model, loader):
    model.eval()
    all_preds = []
    all_ids = []
    print("Master, proses inferensi sedang berjalan...")
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(CFG['device'])
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(ids)
    return all_ids, all_preds

# --- PROSES UTAMA ---

# A. Inisialisasi Model
model = FaceModel(CFG['model_name'], CFG['num_classes']).to(CFG['device'])

# B. Load Data (Pastikan path ini benar sesuai workspace Master)
CSV_PATH = os.path.join('data', 'samplesubmission.csv')
TEST_DIR = os.path.join('data', 'test')

if os.path.exists(CSV_PATH):
    test_df = pd.read_csv(CSV_PATH)
    test_dataset = FaceDataset(test_df, img_dir=TEST_DIR, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=CFG['batch_size'], shuffle=False)

    # C. Jalankan Prediksi
    ids, labels = predict(model, test_loader)

    # D. Simpan ke CSV
    submission = pd.DataFrame({
        'id': ids,
        'label': labels
    })

    submission.to_csv('submission_final.csv', index=False)
    print(f"Selesai Master! File 'submission_final.csv' telah dibuat di {os.getcwd()}")
else:
    print(f"Error: File {CSV_PATH} tidak ditemukan. Master yakin foldernya sudah benar?")