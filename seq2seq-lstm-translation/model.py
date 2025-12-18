import torch
import torch.nn as nn
from data import (TranslationDataset, collate_fn,SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, src_pad_idx, tgt_pad_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"src--source, tgt--target"
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, dropout, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=src_pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):#x=(batch_size,seq_length)
        x = self.embedding(x)#(batch_size,seq_length,emb)
        x = self.dropout(x)
        outputs, (h_t, c_t) = self.lstm(x)
        return h_t, c_t #(num_layers, batch_size, hidden_size)

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, dropout, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=tgt_pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, h_t, c_t):
        if x.dim() == 1:
            x = x.unsqueeze(1) #(batch_size,1)
        x = self.embedding(x)
        x = self.dropout(x)
        outputs, (h_t, c_t) = self.lstm(x, (h_t, c_t))
        logits = self.fc(outputs)
        return logits, (h_t, c_t)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        h_t, c_t = self.encoder(src)
        outputs, _ = self.decoder(tgt[:, :-1], h_t, c_t)  # tgt without <eos> for teacher forcing
        return outputs

    @torch.no_grad()
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, device=device):
        self.eval()
        src = src.to(device)
        batch_size = src.size(0)
        h_t, c_t = self.encoder(src)

        # Decoder input at time-step (0) must be <sos> for every sequence in the batch.
        dec_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        outputs = []

        for _ in range(max_len):
            logits, (h_t, c_t) = self.decoder(dec_input, h_t, c_t)
            pred = logits.argmax(-1).squeeze(1)  # (batch_size,)
            outputs.append(pred)
            if (pred == eos_idx).all():
                break
            dec_input = pred.unsqueeze(1)

        return torch.stack(outputs, dim=1)  # (batch_size, seq_len)