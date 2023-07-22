import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

# Define the Transformer model
class BeeTransformer(nn.Module):
    def __init__(self, num_classes):
        super(BeeTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Define the dataset
class BeeDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

directory_path = "/home/ashdod-ai1/BEE_research/Concatenated"

# Load the concatenated data from the centralized CSV file
concat_file_path = os.path.join(directory_path, "centralized.csv")
data = []
labels = []

with open(concat_file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        if row[1] != 'iteration 3' and row[1] != 'iteration 4' and row[1] != 'iteration 5' and row[1] != 'iteration 6' and row[1] != 'iteration 1' and row[1] != 'iteration 2' :
            if row[0]=='diet 1.1':
                data.append(row[4])  # Modify the index based on the column position of the data in your CSV
                labels.append(row[0])  # Modify the index based on the column position of the label in your CSV
            if row[0]=='diet 5.1':
                data.append(row[4])
                labels.append(row[0])

print(data)
print(labels)

# Initialize the label encoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the dataset
dataset = BeeDataset(data, labels_encoded, tokenizer)

# Split the dataset into training and validation sets
val_size = int(len(dataset) * 0.2)
train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = BeeTransformer(num_classes=len(label_encoder.classes_))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 200

for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            val_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {correct / len(val_dataset):.4f}')

# Calculate the confusion matrix
all_predicted_labels = []
all_true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)

        _, predicted = torch.max(outputs, dim=1)

        all_predicted_labels.extend(predicted.tolist())
        all_true_labels.extend(labels.tolist())

confusion = confusion_matrix(all_true_labels, all_predicted_labels)

# Print the confusion matrix
print('Confusion Matrix:')
print(confusion)

# Extract true positives, true negatives, false positives, false negatives
tn, fp, fn, tp = confusion.ravel()

# Print true positives, true negatives, false positives, false negatives
print('True Positives:', tp)
print('True Negatives:', tn)
print('False Positives:', fp)
print('False Negatives:', fn)