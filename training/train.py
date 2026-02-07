import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainingArguments
print(TrainingArguments.__init__.__code__.co_varnames)

# 1️⃣ Load CSV
df = pd.read_csv("data/raw/sentiment_dataset.csv")

# 2️⃣ Convert sentiment to numeric labels
# Assuming "positive" -> 1, "negative" -> 0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 3️⃣ Split into train and test (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4️⃣ Convert pandas DataFrame to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['text','label']])
test_dataset = Dataset.from_pandas(test_df[['text','label']])

# 5️⃣ Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 6️⃣ Tokenize function
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 7️⃣ Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 8️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # safe for laptop
    logging_dir="./logs",
    save_total_limit=1
)

# 9️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 10️⃣ Train
trainer.train()

# 11️⃣ Save model and tokenizer
model.save_pretrained("model")
tokenizer.save_pretrained("model")
