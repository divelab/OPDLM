from datasets import load_dataset
import json
from collections import Counter
from tqdm import tqdm

DATASET = "open-thoughts/OpenThoughts3-1.2M"
OUTPUT = "data/OpenThoughts3.json"


def get_question(row):
    for message in row["conversations"]:
        role = message.get("from", message.get("role", "")).lower()
        content = message.get("value", message.get("content"))

        if role in ["human", "user"]:
            return content.strip()

    raise ValueError("No human/user message found")


dataset = load_dataset(DATASET, split="train")

output = []
counts = Counter()

for row in tqdm(dataset, desc="Processing OpenThoughts3", unit="examples"):
    question = get_question(row)

    domain = row.get("domain", "unknown")

    output.append({
        "question": question,
        "domain": domain,
    })

    counts[domain] += 1


with open(OUTPUT, "w") as f:
    json.dump(output, f, ensure_ascii=False)

print(f"Wrote {len(output):,} examples to {OUTPUT}")
print("Domains:", counts)

print("\nExample:")
print(json.dumps(output[0], indent=2)[:2000])