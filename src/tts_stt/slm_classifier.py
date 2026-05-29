"""ALAS — SLM intent classifier (tiny NLP).

Classifies the intent of a sentence coming from STT:
  - "system_command"  -> commands that control the program (kapat, dur, çıkış)
  - "navigation"      -> navigation queries (eczane bul, neredeyim)
  - "general"         -> general speech, sentences to be ignored

Approach: hybrid (rule layer + ML model)
  1. Short commands (1-2 words) -> rule-based match (exact result)
  2. Longer sentences -> TF-IDF + LogisticRegression (context analysis)
  3. Confidence threshold -> returns "general" when unsure

Example:
    from tts_stt.slm_classifier import SLMClassifier

    clf = SLMClassifier()
    intent, confidence = clf.predict("kapat")
    # ("system_command", 1.0)  -> caught by the rule layer

    intent, confidence = clf.predict("kapıyı kapat")
    # ("general", 0.87)  -> classified by the ML model
"""

import json
import os
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

try:
    from tts_stt.voice_config import VoiceConfig
except ImportError:
    from voice_config import VoiceConfig

# ──────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ──────────────────────────────────────────────────────────────────────────────

_FILE_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _FILE_DIR / "slm_model.pkl"
_DATA_PATH = _FILE_DIR / "slm_data.json"

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# Sourced from VoiceConfig so confidence tuning has a single source of truth.
DEFAULT_CONFIDENCE_THRESHOLD = VoiceConfig().slm_confidence_threshold

# ──────────────────────────────────────────────────────────────────────────────
# RULE LAYER
# Exact matching for short commands (1-2 words). The ML model struggles with
# these short words due to character similarity; the rule layer fixes that.
# The keyword sets below are the user's spoken Turkish vocabulary (functional).
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
    """Rule-based classification.

    Returns (intent, 1.0) on a match, or None (falls through to the ML model).
    """
    words = text.strip().lower().split()

    if not words:
        return ("general", 1.0)

    # Single word -> exact match.
    if len(words) == 1:
        if words[0] in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)
        return None

    # Two words -> pattern analysis (example phrases kept in Turkish).
    if len(words) == 2:
        first, second = words

        # "tamam kapat", "tamam dur", "tamam çık" -> system_command
        if first == "tamam" and second in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)

        # "artık yeter", "dur artık", "kapat artık" -> system_command
        if second == "artık" and first in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)
        if first == "artık" and second in EXACT_SYSTEM_COMMANDS:
            return ("system_command", 1.0)

        # "navigasyonu kapat", "sesi durdur" -> system_command
        if first in SYSTEM_PREFIXES and second in GENERAL_OBJECT_SUFFIXES:
            return ("system_command", 1.0)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_training_data(data_path: str | None = None) -> list[tuple[str, str]]:
    """Load the training data from slm_data.json."""
    path = data_path or str(_DATA_PATH)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "Place slm_data.json in the src/tts_stt/ folder."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    training_data = []
    for label in ["system_command", "navigation", "general"]:
        sentences = raw.get(label, [])
        for sentence in sentences:
            training_data.append((sentence, label))

    print(f"[SLM] Data loaded: {len(training_data)} sentences "
          f"({sum(1 for _, l in training_data if l == 'system_command')} commands, "
          f"{sum(1 for _, l in training_data if l == 'navigation')} navigation, "
          f"{sum(1 for _, l in training_data if l == 'general')} general)")

    return training_data


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_model(data_path: str | None = None, save_path: str | None = None) -> Pipeline:
    """Train and save the TF-IDF + LogisticRegression model."""
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
    print(f"[SLM] Model saved: {save_to}")

    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

class SLMClassifier:
    """Hybrid intent classifier: rule layer + ML model."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.threshold = confidence_threshold
        path = model_path or str(_MODEL_PATH)

        if not os.path.isfile(path):
            print("[SLM] Model not found, training...")
            self.pipeline = train_model()
        else:
            with open(path, "rb") as f:
                self.pipeline = pickle.load(f)
            print(f"[SLM] Model loaded ✓ (threshold: {self.threshold})")

    def predict(self, text: str) -> tuple[str, float]:
        """Predict the intent of a sentence.

        Flow:
          1. Rule layer -> returns immediately on an exact match
          2. ML model -> classify with TF-IDF + LogReg
          3. Threshold check -> returns "general" when unsure

        Returns
        -------
        (intent, confidence) : ("system_command" | "navigation" | "general", 0.0-1.0)
        """
        text = text.strip().lower()

        if not text:
            return ("general", 1.0)

        # 1. Rule layer.
        rule_result = _rule_based_predict(text)
        if rule_result is not None:
            return rule_result

        # 2. ML model.
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_idx = proba.argmax()
        intent = classes[best_idx]
        confidence = float(proba[best_idx])

        # 3. Threshold check.
        if confidence < self.threshold:
            return ("general", confidence)

        return (intent, confidence)

    def retrain(self, data_path: str | None = None):
        """Re-train the model after slm_data.json has been updated."""
        self.pipeline = train_model(data_path)
        print("[SLM] Model re-trained ✓")


# ──────────────────────────────────────────────────────────────────────────────
# TEST & TRAINING
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  ALAS SLM Classifier v1.2 - Hybrid (Rule + ML)")
    print("=" * 70)

    # Delete the old model and re-train.
    if _MODEL_PATH.exists():
        _MODEL_PATH.unlink()
        print("[SLM] Old model deleted, re-training...")

    model = train_model()
    clf = SLMClassifier()

    # Test sentences (Turkish inputs kept on purpose).
    test_cases = [
        # ── expected: system_command ──
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
        # ── expected: navigation ──
        ("eczane bul",          "navigation"),
        ("en yakın eczane",     "navigation"),
        ("market nerede",       "navigation"),
        ("atm",                 "navigation"),
        ("neredeyim",           "navigation"),
        ("ilaç lazım",          "navigation"),
        ("para çekmem lazım",   "navigation"),
        # ── expected: general ──
        ("kapıyı kapat",        "general"),
        ("pencereyi kapat",     "general"),
        ("arabayı durdur",      "general"),
        ("bugün hava güzel",    "general"),
        ("başım ağrıyor",       "general"),
        ("kutuyu kapat",        "general"),
        ("televizyonu kapat",   "general"),
        # ── ambiguous ──
        ("asdfghjkl",           "general"),
        ("hmm",                 "general"),
    ]

    print(f"\nConfidence threshold: {clf.threshold}")
    print(f"\n{'Sentence':<25} {'Expected':<18} {'Predicted':<18} {'Conf':<8} {'Result'}")
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
            note = " (below threshold)"
        print(f"{text:<25} {expected:<18} {intent:<18} {conf:<8.2f} {match}{note}")

    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.0f}%)")