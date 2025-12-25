import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True,
                            dropout=(dropout if n_layers > 1 else 0), batch_first=True)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W_a = nn.Linear(hid_dim, hid_dim)              
        self.U_a = nn.Linear(2 * hid_dim, hid_dim)          
        self.v_a = nn.Linear(hid_dim, 1)                    
    
    def forward(self, encoder_outputs, decoder_hidden):
        query = self.W_a(decoder_hidden[-1].unsqueeze(1))  
        keys = self.U_a(encoder_outputs)                   
        energy = torch.tanh(query + keys)
        scores = self.v_a(energy).squeeze(2)               
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim)
        self.lstm = nn.LSTM(emb_dim + 2 * hid_dim, hid_dim, n_layers,
                            dropout=(dropout if n_layers > 1 else 0), batch_first=True)
        self.fc = nn.Linear(hid_dim, output_vocab_size)
    
    def forward(self, input_token, hidden, cell, encoder_outputs):
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        context, attn_weights = self.attention(encoder_outputs, hidden)
        context = context.unsqueeze(1)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, teacher_force_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.fc.out_features).to(src.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        n_layers = hidden.size(0) // 2
        hidden = hidden.view(n_layers, 2, hidden.size(1), hidden.size(2)).sum(dim=1)
        cell = cell.view(n_layers, 2, cell.size(1), cell.size(2)).sum(dim=1)
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_force_ratio
            input = tgt[:, t] if teacher_force else output.argmax(1)
        
        return outputs
