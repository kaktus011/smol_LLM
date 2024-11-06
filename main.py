import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizerFast  # For tokenization

# Step 1: Define the Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=4, hidden_size=512, num_layers=4):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer_encoder(embedded)
        return self.fc(output)

# Step 2: Dataset and DataLoader setup
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")['input_ids'].squeeze()
        labels = input_ids.clone()
        return input_ids, labels

def load_data(data_path, tokenizer, max_len):
    with open(data_path, 'r') as f:
        texts = f.readlines()
    return TextDataset(texts, tokenizer, max_len)

# Step 3: Training the model
def train_model(model, data_loader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_ids, labels in data_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

# Step 4: Generate text (inference)
def generate_text(model, tokenizer, prompt, max_len, device):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(input_ids)
            next_token_id = outputs[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# Main execution
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    vocab_size = 30522  # Using BERT tokenizer vocabulary size
    embed_size = 256
    num_heads = 4
    hidden_size = 512
    num_layers = 4
    max_len = 50
    num_epochs = 5
    batch_size = 2
    learning_rate = 1e-4

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Load data
    data_path = "./data/financial_data.txt"
    dataset = load_data(data_path, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    model = SimpleTransformer(vocab_size, embed_size, num_heads, hidden_size, num_layers).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, data_loader, optimizer, num_epochs, device)

    # Test text generation
    prompt = "January - Income $5000, Expenses $3000;"
    print("Generated Summary:", generate_text(model, tokenizer, prompt, max_len=30, device=device))