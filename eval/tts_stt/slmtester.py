"""Interactive tester for the fine-tuned SLM intent classifier.

Loads the fine-tuned MLX model from src/tts_stt/my_custom_slm and classifies
typed commands interactively. The prompt template is kept in Turkish on purpose:
it must match the format the model saw during fine-tuning.

How to run (from the repository root):
    python3 eval/tts_stt/slmtester.py
"""

from pathlib import Path

from mlx_lm import load, generate

VALID_INTENTS = ["system_command", "navigation", "general"]

# The model lives with the source package, not in eval/.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
model_path = str(_REPO_ROOT / "src" / "tts_stt" / "my_custom_slm")

print("Loading model, please wait...")
model, tokenizer = load(model_path)
print("Model loaded successfully!\n")

while True:
    command = input("Enter a command for the glasses (press 'q' to quit): ")
    if command.lower() == "q":
        break

    # Test template matching exactly the format the model saw during training
    # (kept in Turkish on purpose — do not translate).
    prompt = (
        "<|im_start|>user\n"
        "Aşağıdaki komutun niyetini (intent) sınıflandır. "
        "Seçenekler: 'system_command', 'navigation', 'general'.\n"
        f"Komut: {command}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Generate the model's answer.
    answer = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    # Clean up: strip extra tokens and keep only the first word.
    intent = answer.strip().lower()
    intent = intent.replace("<|im_end|>", "").strip()
    intent = intent.split()[0] if intent else "unknown"

    # Validity check.
    if intent in VALID_INTENTS:
        print(f"AI says: {intent}\n")
    else:
        print(f"[warn] Unexpected output: '{intent}' -> fallback: general\n")
