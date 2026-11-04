
import csv
import random
from pathlib import Path

random.seed(45)

samples = [
    ("Your order has been shipped.", 0.95),
    ("Your package will arrive tomorrow.", 0.95),
    ("We have processed your refund.", 0.90),
    ("Please contact support for help.", 0.85),
    ("Your payment was successful.", 0.95),
    ("We cannot help you with this.", 0.40),
    ("Maybe your order will come.", 0.45),
    ("I don't know anything.", 0.10),
    ("Your shipment is delayed.", 0.90),
    ("Thank you for contacting us.", 0.85),
    ("Delivered", 0.99)
]

variations = [
    ("pls", "please"),
    ("pkg", "package"),
    ("tmrw", "tomorrow"),
    ("thx", "thank you"),
    ("ur", "your"),
]

rows = []

for text, score in samples:
    rows.append((text, score))

    for short, full in variations:
        if short in text.lower():
            modified = text.lower().replace(short, full)
            rows.append((modified, score))

            noisy = text.lower().replace(short, short.upper())
            rows.append((noisy, score))

# Synthetic examples are for demonstration only.
# Real production training needs human-reviewed labels.

output = Path("data/raw/responses.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "score"])
    writer.writerows(rows)

print(f"Generated {len(rows)} examples")
