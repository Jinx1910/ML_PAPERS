import torch
import torch.nn as nn
import torch.optim as optim

from model import Encoder, Decoder, Seq2Seq
from data import SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, tgt_pad_idx, train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embedding_size = 256
hidden_size = 512
dropout = 0.5
num_layers = 2
batch_size = 32
epochs = 10
lr = 0.001

encoder = Encoder(SRC_VOCAB_SIZE, embedding_size, dropout, hidden_size, num_layers)
decoder = Decoder(TGT_VOCAB_SIZE, embedding_size, dropout, hidden_size, num_layers)
model = Seq2Seq(encoder, decoder).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=lr)

def train_model(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        outputs = model(src, tgt)  #(batch, seq_len, vocab_size)
        loss = loss_fn(outputs.view(-1, TGT_VOCAB_SIZE), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt)
            loss = loss_fn(outputs.view(-1, TGT_VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


print("Starting training...\n")

for epoch in range(1, epochs + 1):
    train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)

    print(f"Epoch {epoch}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print("-" * 40)

print("Training complete!")

torch.save(model.state_dict(), "seq2seq_translation_model.pth")
print("Model saved as 'seq2seq_translation_model.pth'")