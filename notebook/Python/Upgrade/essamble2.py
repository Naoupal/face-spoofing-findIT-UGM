import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
from facenet_pytorch import MTCNN
from tqdm import tqdm

def run_ensemble():
    # --- KONFIGURASI PATH ABSOLUT (Sesuai Struktur Folder Master) ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = r'E:\Python - Project\face-spoofing-findIT-UGM'
    
    TEST_DIR = os.path.join(BASE_DIR, 'data', 'test') # Folder gambar test
    SAMPLE_CSV = os.path.join(BASE_DIR, 'outputs', 'samplesubmission.csv')
    SAVE_PATH = os.path.join(BASE_DIR, 'submission', 'submission_ensemble_final.csv')
    
    # Path Model di folder outputs
    MODEL_EFF_PATH = os.path.join(BASE_DIR, 'outputs', 'efficientnet_b3_model.pth')
    MODEL_CONV_PATH = os.path.join(BASE_DIR, 'outputs', 'best_convnext_tiny.pth')

    CLASS_NAMES = ['fake_mannequin', 'fake_mask', 'fake_printed', 'fake_screen', 'fake_unknown', 'realperson']

    print(f"--- Menggunakan Device: {DEVICE} ---")

    # 1. Transformasi
    transform_eff = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_conv = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load MTCNN & Models
    mtcnn = MTCNN(keep_all=False, select_largest=True, margin=60, post_process=False, device=DEVICE)

    print("Loading EfficientNet-B3...")
    model_eff = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(CLASS_NAMES))
    model_eff.load_state_dict(torch.load(MODEL_EFF_PATH, map_location=DEVICE))
    model_eff = model_eff.to(DEVICE).eval()

    print("Loading ConvNeXt-Tiny...")
    model_conv = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(CLASS_NAMES))
    model_conv.load_state_dict(torch.load(MODEL_CONV_PATH, map_location=DEVICE))
    model_conv = model_conv.to(DEVICE).eval()

    # 3. Inference
    submission_df = pd.read_csv(SAMPLE_CSV)
    predictions = []

    print(f"🚀 Memulai Prediksi Ensemble pada {len(submission_df)} gambar...")
    
    with torch.no_grad():
        for img_id in tqdm(submission_df['id']):
            # Cari file gambar (mendukung berbagai ekstensi .jpg, .png, dsb)
            search_pattern = os.path.join(TEST_DIR, f"{img_id}.*")
            matching_files = glob.glob(search_pattern)
            
            if not matching_files:
                predictions.append('fake_unknown')
                continue
                
            img_path = matching_files[0]
            
            try:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)
                
                # Smart Fallback jika wajah tidak terdeteksi
                if face is not None:
                    pil_img = transforms.ToPILImage()(face.byte())
                else:
                    w, h = img.size
                    pil_img = img.crop((w*0.2, h*0.2, w*0.8, h*0.8)) # Center Crop
                
                # Siapkan Tensor
                t_eff = transform_eff(pil_img).unsqueeze(0).to(DEVICE)
                t_conv = transform_conv(pil_img).unsqueeze(0).to(DEVICE)
                
                # Prediksi (Weighted 0.35 : 0.65)
                prob_eff = F.softmax(model_eff(t_eff), dim=1)
                prob_conv = F.softmax(model_conv(t_conv), dim=1)
                
                avg_prob = (prob_eff * 0.35) + (prob_conv * 0.65)
                
                _, predicted_idx = torch.max(avg_prob, 1)
                predictions.append(CLASS_NAMES[predicted_idx.item()])
                
            except Exception as e:
                predictions.append('fake_unknown')

    # 4. Simpan ke CSV
    submission_df['label'] = predictions
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    submission_df.to_csv(SAVE_PATH, index=False)
    
    print("\n" + "="*40)
    print(f"✅ Submission Berhasil! Lokasi: {SAVE_PATH}")
    print("="*40)
    print(submission_df['label'].value_counts())

if __name__ == '__main__':
    run_ensemble()