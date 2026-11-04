
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from torch.utils.data import Dataset

MODEL_NAME = "distilbert-base-uncased"

df = pd.read_csv("data/raw/responses.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ResponseDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["score"].astype(float).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.float32,
        )

        return item

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
)

train_dataset = ResponseDataset(train_df)
val_dataset = ResponseDataset(val_df)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    mae = mean_absolute_error(labels, predictions)

    rmse = np.sqrt(
        mean_squared_error(labels, predictions)
    )

    return {
        "mae": mae,
        "rmse": rmse,
    }

args = TrainingArguments(
    output_dir="artifacts/checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_mae",
    greater_is_better=False,
    logging_steps=5,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate()

print("Evaluation:", metrics)

trainer.save_model("artifacts/best_model")
tokenizer.save_pretrained("artifacts/best_model")

print("Saved best model")
