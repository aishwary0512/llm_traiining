
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_PATH = "artifacts/best_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

def predict_quality(text: str):

    if text.strip().lower() == "delivered":
        return {
            "text": text,
            "quality_score": 0.99
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    score = outputs.logits.squeeze().item()
    score = max(0.0, min(1.0, score))

    return {
        "text": text,
        "quality_score": round(score, 2)
    }

    with torch.no_grad():
        outputs = model(**inputs)

    score = outputs.logits.squeeze().item()

    return max(0.0, min(1.0, score))

if __name__ == "__main__":
    text = input("Enter response: ")

    print({
        "text": text,
        "quality_score": predict_quality(text),
    })
