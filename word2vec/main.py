import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()

text = load_text("file.txt")
tokens = text.split()

def build_vocab(tokens):
    vocab = sorted(set(tokens))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos

stoi, itos = build_vocab(tokens)
vocab_size = len(stoi)

class CBOWDataset(Dataset):
    def __init__(self, tokens, stoi, window):
        self.data = []
        for i in range(window, len(tokens) - window):
            context = tokens[i-window:i] + tokens[i+1:i+window+1]
            target = tokens[i]
            self.data.append(([stoi[w] for w in context], stoi[target]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long),torch.tensor(target, dtype=torch.long)


class SkipGramDataset(Dataset):
    def __init__(self, tokens, stoi, window):
        self.data = []
        for i in range(window, len(tokens) - window):
            center = stoi[tokens[i]]
            context = tokens[i-window:i] + tokens[i+1:i+window+1]
            for w in context:
                self.data.append((center, stoi[w]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        return torch.tensor(center, dtype=torch.long),torch.tensor(context, dtype=torch.long)

class SimpleCBOW(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.linear = nn.Linear(embedding_dim,vocab_size)
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, context):
        emb = self.embeddings(context)      # (B, 2N, D)
        avg = emb.mean(dim=1)               # (B, D)
        return self.linear(avg)                # (B, V)

    def get_embeddings(self):
        return self.embeddings.weight.detach().cpu().numpy()


class SimpleSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, center):
        emb = self.embeddings(center)       # (B, D)
        return self.linear(emb)             # (B, V)

    def get_embeddings(self):
        return self.embeddings.weight.detach().cpu().numpy()


USE_CBOW = True        #False for Skip-gram
window_size = 2
embedding_dim = 50
batch_size = 16
epochs = 200
lr= 1e-3


if USE_CBOW:
    dataset = CBOWDataset(tokens, stoi, window_size)
    model = SimpleCBOW(vocab_size, embedding_dim).to(device)
else:
    dataset = SkipGramDataset(tokens, stoi, window_size)
    model = SimpleSkipGram(vocab_size, embedding_dim).to(device)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    total_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Avg Loss {avg_loss:.4f}")

embeddings = model.get_embeddings()
