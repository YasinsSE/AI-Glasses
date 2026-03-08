"""
ALAS - SLM Classifier (Tiny NLP)
=======================================
STT'den gelen cümlenin niyetini sınıflandırır:
  - "system_command"  → Programı kontrol eden komutlar (kapat, dur, çıkış)
  - "navigation"      → Navigasyon sorgusu (eczane bul, neredeyim)
  - "general"         → Genel konuşma, yok sayılacak cümleler

Yaklaşım: Hibrit (Kural Katmanı + ML Modeli)
  1. Kısa komutlar (1-2 kelime) → kural tabanlı eşleşme (kesin sonuç)
  2. Uzun cümleler → TF-IDF + LogisticRegression (bağlam analizi)
  3. Confidence threshold → emin değilse "general" döner

Kullanım:
    from tts_stt.slm_classifier import SLMClassifier

    clf = SLMClassifier()
    intent, confidence = clf.predict("kapat")
    # ("system_command", 1.0)  → kural katmanı yakaladı

    intent, confidence = clf.predict("kapıyı kapat")
    # ("general", 0.87)  → ML modeli sınıflandırdı
"""

import json
import os
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ──────────────────────────────────────────────────────────────────────────────
# DOSYA YOLLARI
# ──────────────────────────────────────────────────────────────────────────────

_FILE_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _FILE_DIR / "slm_model.pkl"
_DATA_PATH = _FILE_DIR / "slm_data.json"

# ──────────────────────────────────────────────────────────────────────────────
# VARSAYILAN AYARLAR
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# ──────────────────────────────────────────────────────────────────────────────
# KURAL KATMANI
# Kısa komutlar (1-2 kelime) için kesin eşleşme.
# ML modeli bu kısa kelimelerde karakter benzerliği yüzünden zorlanıyor,
# kural katmanı bu sorunu çözer.
# ──────────────────────────────────────────────────────────────────────────────

EXACT_SYSTEM_COMMANDS = {
    "kapat", "çıkış", "çık", "dur", "durdur", "iptal",
    "yeter", "sus", "bitir", "bitti", "sonlandır",
    "bırak", "vazgeç",
}

SYSTEM_PREFIXES = {
    "navigasyonu", "programı", "uygulamayı", "sistemi",
    "sesi", "rotayı", "yönlendirmeyi", "sesli",
    "hepsini", "her",
}

GENERAL_OBJECT_SUFFIXES = {"kapat", "kapa", "durdur"}


def _rule_based_predict(text: str) -> tuple[str, float] | None:
    """
    Kural tabanlı sınıflandırma. Eşleşme varsa (intent, 1.0) döner.
    Eşleşme yoksa None döner → ML modeline düşer.
    """
    words = text.strip().lower().split()

    if not words:
        return ("general", 1.0)

    # Tek kelime → exact match
    if len(words) == 1:
        if words[0] in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)
        return None

    # İki kelime → kalıp analizi
    if len(words) == 2:
        first, second = words

        # "tamam kapat", "tamam dur", "tamam çık" → system_command
        if first == "tamam" and second in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)

        # "artık yeter", "dur artık", "kapat artık" → system_command
        if second == "artık" and first in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)
        if first == "artık" and second in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)

        # "navigasyonu kapat", "sesi durdur" → system_command
        if first in SYSTEM_PREFIXES and second in GENERAL_OBJECT_SUFFIXES:
            return ("system_command", 1.0)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ──────────────────────────────────────────────────────────────────────────────

def load_training_data(data_path: str | None = None) -> list[tuple[str, str]]:
    """slm_data.json dosyasından eğitim verisini yükler."""
    path = data_path or str(_DATA_PATH)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Eğitim verisi bulunamadı: {path}\n"
            "slm_data.json dosyasını src/tts_stt/ klasörüne koyun."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    training_data = []
    for label in ["system_command", "navigation", "general"]:
        sentences = raw.get(label, [])
        for sentence in sentences:
            training_data.append((sentence, label))

    print(f"[SLM] Veri yüklendi: {len(training_data)} cümle "
          f"({sum(1 for _, l in training_data if l == 'system_command')} komut, "
          f"{sum(1 for _, l in training_data if l == 'navigation')} navigasyon, "
          f"{sum(1 for _, l in training_data if l == 'general')} genel)")

    return training_data


# ──────────────────────────────────────────────────────────────────────────────
# EĞİTİM
# ──────────────────────────────────────────────────────────────────────────────

def train_model(data_path: str | None = None, save_path: str | None = None) -> Pipeline:
    """TF-IDF + LogisticRegression modeli eğitir ve kaydeder."""
    training_data = load_training_data(data_path)
    texts = [t for t, _ in training_data]
    labels = [l for _, l in training_data]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    pipeline.fit(texts, labels)

    scores = cross_val_score(pipeline, texts, labels, cv=3, scoring="accuracy")
    print(f"[SLM] Cross-val accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

    save_to = save_path or str(_MODEL_PATH)
    with open(save_to, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[SLM] Model kaydedildi: {save_to}")

    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# SINIFLANDIRICI
# ──────────────────────────────────────────────────────────────────────────────

class SLMClassifier:
    """Hibrit intent sınıflandırıcı: kural katmanı + ML modeli."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.threshold = confidence_threshold
        path = model_path or str(_MODEL_PATH)

        if not os.path.isfile(path):
            print("[SLM] Model bulunamadı, eğitiliyor...")
            self.pipeline = train_model()
        else:
            with open(path, "rb") as f:
                self.pipeline = pickle.load(f)
            print(f"[SLM] Model yüklendi ✓ (threshold: {self.threshold})")

    def predict(self, text: str) -> tuple[str, float]:
        """
        Cümlenin niyetini tahmin et.

        Akış:
          1. Kural katmanı → kesin eşleşme varsa direkt döner
          2. ML modeli → TF-IDF + LogReg ile sınıflandır
          3. Threshold kontrolü → emin değilse "general" döner

        Returns
        -------
        (intent, confidence) : ("system_command" | "navigation" | "general", 0.0-1.0)
        """
        text = text.strip().lower()

        if not text:
            return ("general", 1.0)

        # 1. Kural katmanı
        rule_result = _rule_based_predict(text)
        if rule_result is not None:
            return rule_result

        # 2. ML modeli
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_idx = proba.argmax()
        intent = classes[best_idx]
        confidence = float(proba[best_idx])

        # 3. Threshold kontrolü
        if confidence < self.threshold:
            return ("general", confidence)

        return (intent, confidence)

    def retrain(self, data_path: str | None = None):
        """slm_data.json güncellendikten sonra modeli yeniden eğit."""
        self.pipeline = train_model(data_path)
        print("[SLM] Model yeniden eğitildi ✓")


# ──────────────────────────────────────────────────────────────────────────────
# TEST & EĞİTİM
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  ALAS SLM Classifier v1.2 - Hibrit (Kural + ML)")
    print("=" * 70)

    # Eski modeli sil ve yeniden eğit
    if _MODEL_PATH.exists():
        _MODEL_PATH.unlink()
        print("[SLM] Eski model silindi, yeniden eğitiliyor...")

    model = train_model()
    clf = SLMClassifier()

    # Test cümleleri
    test_cases = [
        # ── system_command beklenen ──
        ("kapat",               "system_command"),
        ("programı kapat",      "system_command"),
        ("navigasyonu durdur",  "system_command"),
        ("çıkış",               "system_command"),
        ("sesi kapat",          "system_command"),
        ("yeter",               "system_command"),
        ("bitir",               "system_command"),
        ("dur",                 "system_command"),
        ("tamam kapat",         "system_command"),
        ("artık yeter",         "system_command"),
        # ── navigation beklenen ──
        ("eczane bul",          "navigation"),
        ("en yakın eczane",     "navigation"),
        ("market nerede",       "navigation"),
        ("atm",                 "navigation"),
        ("neredeyim",           "navigation"),
        ("ilaç lazım",          "navigation"),
        ("para çekmem lazım",   "navigation"),
        # ── general beklenen ──
        ("kapıyı kapat",        "general"),
        ("pencereyi kapat",     "general"),
        ("arabayı durdur",      "general"),
        ("bugün hava güzel",    "general"),
        ("başım ağrıyor",       "general"),
        ("kutuyu kapat",        "general"),
        ("televizyonu kapat",   "general"),
        # ── belirsiz ──
        ("asdfghjkl",           "general"),
        ("hmm",                 "general"),
    ]

    print(f"\nConfidence threshold: {clf.threshold}")
    print(f"\n{'Cümle':<25} {'Beklenen':<18} {'Tahmin':<18} {'Güven':<8} {'Sonuç'}")
    print("-" * 80)

    correct = 0
    total = len(test_cases)

    for text, expected in test_cases:
        intent, conf = clf.predict(text)
        match = "✓" if intent == expected else "✗"
        if intent == expected:
            correct += 1
        note = ""
        if conf < clf.threshold:
            note = " (threshold altı)"
        print(f"{text:<25} {expected:<18} {intent:<18} {conf:<8.2f} {match}{note}")

    print(f"\nDoğruluk: {correct}/{total} ({correct/total*100:.0f}%)")