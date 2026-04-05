import pandas as pd

# 1. Baca file yang sudah Master buat
file_name = 'submission_final.csv'
df = pd.read_csv(file_name)

# 2. Logika Convert: 
# Jika label adalah 5 (Realperson), ubah jadi 1.
# Jika label adalah 0, 1, 2, 3, atau 4 (Semua jenis Fake), ubah jadi 0.
def convert_to_binary(label):
    if label == 5:
        return 1
    else:
        return 0

df['label'] = df['label'].apply(convert_to_binary)

# 3. Simpan kembali ke file baru
df.to_csv('submission_final_fixed.csv', index=False)

print("✅ Master, file 'submission_final_fixed.csv' telah siap!")
print("Isi label sekarang hanya 0 (Fake) dan 1 (Real).")