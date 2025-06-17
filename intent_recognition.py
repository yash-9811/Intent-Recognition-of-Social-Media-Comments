import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset
df = pd.read_excel(r'C:\Users\yashs\Desktop\Major_Data_Set_Sample.xlsx')

# Ensure all comments are strings
df['Comment'] = df['Comment'].astype(str)

comments = df['Comment'].tolist()
intents = df['Intent'].tolist()

# Preprocess the data
label_encoder = LabelEncoder()
filtered_comments = []
filtered_intents = []

for comment, intent in zip(comments, intents):
    if not any(char in comment for char in ['😊', '...']):
        filtered_comments.append(comment)
        filtered_intents.append(intent)

# Encode the labels
labels = label_encoder.fit_transform(filtered_intents)
num_intent_classes = len(label_encoder.classes_)

# Tokenize the comments using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_input_ids = []
encoded_attention_masks = []

max_seq_length = 128

for comment in filtered_comments:
    encoded = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        return_attention_mask=True,
        truncation=True
    )
    encoded_input_ids.append(encoded['input_ids'])
    encoded_attention_masks.append(encoded['attention_mask'])

input_ids_tensor = torch.tensor(encoded_input_ids)
attention_masks_tensor = torch.tensor(encoded_attention_masks)

# Split the data into training and testing sets
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(
    input_ids_tensor, attention_masks_tensor, labels, random_state=42, test_size=0.2
)

# Create DataLoader for training and testing sets
train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(test_inputs, test_masks, torch.tensor(test_labels))
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Load pre-trained BERT model
pretrained_model = "bert-large-uncased"
model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_intent_classes)

# Fine-tune the pre-trained model
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, label = batch
        logits = model(input_ids, attention_mask=attention_mask)[0]
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Avg. Training Loss: {avg_train_loss:.4f}")
    scheduler.step()

# Evaluate the model
model.eval()
all_preds = []
all_labels = []

for batch in test_dataloader:
    input_ids, attention_mask, label = batch
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)[0]
    preds = torch.argmax(logits, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(label.cpu().numpy())

# Calculate accuracy and classification report
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

target_names = label_encoder.classes_
report = classification_report(all_labels, all_preds, labels=label_encoder.transform(target_names), target_names=target_names, output_dict=True)
print(classification_report(all_labels, all_preds, labels=label_encoder.transform(target_names), target_names=target_names))

# User interaction
def predict_intent_and_confidence(comment):
    if not any(char in comment for char in ['😊', '...']):
        encoded_comment = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            logits = model(encoded_comment['input_ids'], attention_mask=encoded_comment['attention_mask'])[0]
        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        predicted_label = label_encoder.classes_[predicted_class]
        confidence = confidence.item()
        return predicted_label, confidence

while True:
    user_input = input("Enter a comment (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    intent, confidence = predict_intent_and_confidence(user_input)
    print(f"Intent: {intent}, Confidence: {confidence:.4f}")
