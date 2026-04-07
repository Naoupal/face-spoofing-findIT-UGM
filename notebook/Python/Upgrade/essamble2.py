import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import timm
from tqdm import tqdm

# --- Konfigurasi ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE_C = 224
IMG_SIZE_E = 300

TEST_DIR = r'E:\Python - Project\face-spoofing-findIT-UGM\data\test'
SAMPLE_SUB = 'samplesubmission.csv'

MODEL_PATH_C = r'E:\Python - Project\face-spoofing-findIT-UGM\outputs\best_convnext_tiny.pth'
MODEL_PATH_E = r'E:\Python - Project\face-spoofing-findIT-UGM\outputs\efficientnet_b3_model.pth'


# 🔥 FUNCTION UNTUK FIX STATE_DICT
def clean_state_dict(state_dict):
    # kalau disimpan dalam dict {'model': ...}
    if 'model' in state_dict:
        state_dict = state_dict['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        # hapus prefix "model." atau "module."
        new_k = k.replace("model.", "").replace("module.", "")
        new_state_dict[new_k] = v

    return new_state_dict

def safe_load(model, path):
    state_dict = torch.load(path, map_location=DEVICE)

    if 'model' in state_dict:
        state_dict = state_dict['model']

    # bersihin prefix
    new_state = {}
    for k, v in state_dict.items():
        new_k = k.replace("model.", "").replace("module.", "")
        new_state[new_k] = v

    try:
        model.load_state_dict(new_state)
    except Exception as e:
        print(f"❌ ERROR loading {path}")
        print(e)
        exit()

    return model

# --- Load Model ---
def load_models():
    # ConvNeXt
    model_c = timm.create_model('convnext_tiny', pretrained=False, num_classes=6)
    model_c = safe_load(model_c, MODEL_PATH_C)
    model_c.to(DEVICE).eval()

    # EfficientNet
    model_e = timm.create_model('efficientnet_b3', pretrained=False, num_classes=6)
    model_e = safe_load(model_e, MODEL_PATH_E)
    model_e.to(DEVICE).eval()

    return model_c, model_e


# --- Transformasi ---
transform_c = transforms.Compose([
    transforms.Resize((IMG_SIZE_C, IMG_SIZE_C)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_e = transforms.Compose([
    transforms.Resize((IMG_SIZE_E, IMG_SIZE_E)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# --- Label ---
class_names = [
    'fake_mannequin',
    'fake_mask',
    'fake_printed',
    'fake_screen',
    'fake_unknown',
    'realperson'
]


# --- Inference ---
def ensemble_inference():
    model_c, model_e = load_models()

    df_sub = pd.read_csv(SAMPLE_SUB)
    results = []

    print(f"🚀 Ensemble inference: {len(df_sub)} images")

    with torch.no_grad():
        for img_name in tqdm(df_sub['id']):
            img_path = os.path.join(TEST_DIR, f"{img_name}.jpg")

            image = Image.open(img_path).convert('RGB')

            input_c = transform_c(image).unsqueeze(0).to(DEVICE)
            input_e = transform_e(image).unsqueeze(0).to(DEVICE)

            out_c = F.softmax(model_c(input_c), dim=1)
            out_e = F.softmax(model_e(input_e), dim=1)

            # 🔥 weighted ensemble (lebih bagus dari avg biasa)
            avg = (0.6 * out_c + 0.4 * out_e)

            _, pred = torch.max(avg, 1)
            label = class_names[pred.item()]

            results.append(label)

    df_sub['label'] = results
    df_sub.to_csv('submission_ensemble_final.csv', index=False)

    print("✅ DONE: submission_ensemble_final.csv")


if __name__ == "__main__":
    ensemble_inference()