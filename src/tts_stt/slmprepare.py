import json
import os
import random

# Tekrarlanabilirlik için sabit seed
random.seed(42)

# Dosya yolları — scriptin kendi dizinine göre
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "slm_data.json")
output_dir = script_dir

# Masaüstünde klasörü oluştur
os.makedirs(output_dir, exist_ok=True)

# Senin verini okuyoruz
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = ["system_command", "navigation", "general"]
formatted_data = []

# Veriyi Qwen'in anlayacağı ChatML formatına çeviriyoruz
for category in categories:
    if category in data:
        count = len(data[category])
        print(f"📂 {category}: {count} örnek bulundu")
        for sentence in data[category]:
            # ÖNEMLİ: Modele sadece kelimeyi değil, ne yapması gerektiğini de söylüyoruz.
            prompt = f"Aşağıdaki komutun niyetini (intent) sınıflandır. Seçenekler: 'system_command', 'navigation', 'general'.\nKomut: {sentence}"

            # MLX için en güvenilir sohbet (ChatML) formatı
            chat_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{category}<|im_end|>"

            formatted_data.append({"text": chat_text})
    else:
        print(f"⚠️  {category}: veri bulunamadı, atlanıyor")

# Modelin ezberlememesi için verileri rastgele karıştırıyoruz
random.shuffle(formatted_data)

# %85 Eğitim (Train), %15 Test (Valid) olarak ayır
split_index = int(len(formatted_data) * 0.85)
train_data = formatted_data[:split_index]
valid_data = formatted_data[split_index:]

# Train dosyasını kaydet
with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Valid dosyasını kaydet
with open(os.path.join(output_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ İşlem tamam! Toplam {len(formatted_data)} komut dönüştürüldü.")
print(f"Eğitim verisi (Train): {len(train_data)} satır")
print(f"Test verisi (Valid): {len(valid_data)} satır")
print(f"Dosyalar {output_dir} konumuna kaydedildi.")