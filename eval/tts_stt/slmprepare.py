"""Prepare SLM fine-tuning data from the intent dataset.

Reads the labelled intent dataset (src/tts_stt/slm_data.json) and converts it
into ChatML-formatted train/valid JSONL files for MLX LoRA fine-tuning.

    Inputs : src/tts_stt/slm_data.json
    Outputs: outputs/eval/tts_stt/{train.jsonl, valid.jsonl}

How to run (from the repository root):
    python3 eval/tts_stt/slmprepare.py
"""

import json
import os
import random
from pathlib import Path

# Fixed seed for reproducibility.
random.seed(42)

# Resolve paths relative to the repository root so the script works from any
# working directory.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
input_file = _REPO_ROOT / "src" / "tts_stt" / "slm_data.json"
output_dir = _REPO_ROOT / "outputs" / "eval" / "tts_stt"

output_dir.mkdir(parents=True, exist_ok=True)

# Load the labelled dataset.
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = ["system_command", "navigation", "general"]
formatted_data = []

# Convert the data into the ChatML format that Qwen expects.
for category in categories:
    if category in data:
        count = len(data[category])
        print(f"[data] {category}: {count} samples found")
        for sentence in data[category]:
            # Give the model the task, not just the word, to classify the intent.
            prompt = (
                "Classify the intent of the following command. "
                "Options: 'system_command', 'navigation', 'general'.\n"
                f"Command: {sentence}"
            )
            # The most reliable chat (ChatML) format for MLX.
            chat_text = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{category}<|im_end|>"
            )
            formatted_data.append({"text": chat_text})
    else:
        print(f"[warn] {category}: no data found, skipping")

# Shuffle so the model does not learn the category ordering.
random.shuffle(formatted_data)

# Split 85% train / 15% validation.
split_index = int(len(formatted_data) * 0.85)
train_data = formatted_data[:split_index]
valid_data = formatted_data[split_index:]

with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(output_dir / "valid.jsonl", "w", encoding="utf-8") as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n[done] Converted {len(formatted_data)} commands.")
print(f"Train set: {len(train_data)} lines")
print(f"Valid set: {len(valid_data)} lines")
print(f"Files saved to {output_dir}")
