import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import timm
from tqdm import tqdm

# --- Konfigurasi ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE_C = 224  # Size untuk ConvNeXt
IMG_SIZE_E = 300  # Size untuk EfficientNet
TEST_DIR = r'E:\Python - Project\face-spoofing-findIT-UGM\data\test' # Sesuaikan path test Master
SAMPLE_SUB = 'samplesubmission.csv'
# Pastikan path ini benar-benar ada di file explorer Master
MODEL_PATH_C = r'E:\Python - Project\face-spoofing-findIT-UGM\outputs\best_convnext_tiny.pth' 
MODEL_PATH_E = r'E:\Python - Project\face-spoofing-findIT-UGM\outputs\efficientnet_b3_model.pth'

# --- Load Model ---
def load_models():
    # Model C: ConvNeXt-Tiny (Asumsi 6 kelas berdasarkan info sebelumnya)
    model_c = timm.create_model('convnext_tiny', pretrained=False, num_classes=6)
    model_c.load_state_dict(torch.load(MODEL_PATH_C, map_location=DEVICE))
    model_c.to(DEVICE)
    model_c.eval()

    # Model E: EfficientNet-B3
    model_e = timm.create_model('efficientnet_b3', pretrained=False, num_classes=6)
    model_e.load_state_dict(torch.load(MODEL_PATH_E, map_location=DEVICE))
    model_e.to(DEVICE)
    model_e.eval()
    
    return model_c, model_e

# --- Transformasi ---
transform_c = transforms.Compose([
    transforms.Resize((IMG_SIZE_C, IMG_SIZE_C)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_e = transforms.Compose([
    transforms.Resize((IMG_SIZE_E, IMG_SIZE_E)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapping Label (Sesuaikan dengan urutan folder saat training)
class_names = ['fake_mannequin', 'fake_mask', 'fake_printed', 'fake_screen', 'fake_unknown', 'realperson']

def ensemble_inference():
    model_c, model_e = load_models()
    df_sub = pd.read_csv(SAMPLE_SUB)
    results = []

    print(f"🚀 Melakukan Ensemble Inference pada {len(df_sub)} gambar...")

    with torch.no_grad():
        for img_name in tqdm(df_sub['id']):
            # Load Image
            img_path = os.path.join(TEST_DIR, f"{img_name}.jpg") # Sesuaikan ekstensi (.jpg/.png)
            image = Image.open(img_path).convert('RGB')

            # Preprocess untuk masing-masing model
            input_c = transform_c(image).unsqueeze(0).to(DEVICE)
            input_e = transform_e(image).unsqueeze(0).to(DEVICE)

            # Get Probabilities (Soft Voting)
            output_c = F.softmax(model_c(input_c), dim=1)
            output_e = F.softmax(model_e(input_e), dim=1)

            # Rata-rata probabilitas (Bobot bisa diatur, misal 0.5 & 0.5)
            avg_output = (output_c + output_e) / 2
            
            # Ambil index tertinggi
            _, predicted_idx = torch.max(avg_output, 1)
            label = class_names[predicted_idx.item()]
            
            results.append(label)

    # Simpan Hasil
    df_sub['label'] = results
    df_sub.to_csv('submission_ensemble_final.csv', index=False)
    print("✅ Submission berhasil dibuat: submission_ensemble_final.csv")

if __name__ == "__main__":
    ensemble_inference()