
from pathlib import Path
from mlx_lm import load, generate

VALID_INTENTS = ["system_command", "navigation", "general"]

# Scriptin kendi dizinine göre model yolu
model_path = str(Path(__file__).resolve().parent / "my_custom_slm")

print("Model yükleniyor, lütfen bekleyin...")
model, tokenizer = load(model_path)
print("Model başarıyla yüklendi!\n")

while True:
    komut = input("Gözlüğe bir komut söyle (Çıkmak için 'q'): ")
    if komut.lower() == "q":
        break

    # Modelin eğitimde gördüğü formata tam olarak uygun test şablonu
    prompt = (
        "<|im_start|>user\n"
        "Aşağıdaki komutun niyetini (intent) sınıflandır. "
        "Seçenekler: 'system_command', 'navigation', 'general'.\n"
        f"Komut: {komut}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Modelden cevabı üretiyoruz
    cevap = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    # Temizlik: ek token'ları sil, sadece ilk kelimeyi al
    intent = cevap.strip().lower()
    intent = intent.replace("<|im_end|>", "").strip()
    intent = intent.split()[0] if intent else "bilinmiyor"

    # Geçerlilik kontrolü
    if intent in VALID_INTENTS:
        print(f"🤖 Yapay Zeka Diyor Ki: {intent}\n")
    else:
        print(f"⚠️  Beklenmeyen çıktı: '{intent}' → fallback: general\n")