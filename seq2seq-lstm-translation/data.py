import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

df = pd.read_csv('deu.txt',delimiter='\t',header=None)
df.columns = ['English','German','Source']
del df['Source']

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

def tokenize(sentence):
    return sentence.lower().split()

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>","<unk>"]

def build_vocab(sentences):
    words = set()
    for s in sentences:
        words.update(tokenize(s))
    vocab = SPECIAL_TOKENS + sorted(words)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos, len(vocab)


src_stoi, src_itos, SRC_VOCAB_SIZE = build_vocab(train_df["English"])
tgt_stoi, tgt_itos, TGT_VOCAB_SIZE = build_vocab(train_df["German"])

def text_to_indices(text, vocab):
    indices = [vocab["<sos>"]]

    for token in tokenize(text):
        indices.append(vocab.get(token, vocab["<unk>"]))

    indices.append(vocab["<eos>"])
    return indices

src_seqs = [text_to_indices(s, src_stoi) for s in train_df["English"]]
tgt_seqs = [text_to_indices(s, tgt_stoi) for s in train_df["German"]]

class TranslationDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return self.src_seqs[idx], self.tgt_seqs[idx]
    
src_pad_idx = src_stoi["<pad>"]
tgt_pad_idx = tgt_stoi["<pad>"]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_len = max(len(s) for s in src_batch)
    tgt_len = max(len(t) for t in tgt_batch)

    src_padded = torch.tensor([s + [src_pad_idx] * (src_len - len(s)) for s in src_batch])
    tgt_padded = torch.tensor([t + [tgt_pad_idx] * (tgt_len - len(t)) for t in tgt_batch])

    return src_padded, tgt_padded 

train_src = [text_to_indices(s, src_stoi) for s in train_df["English"]]
train_tgt = [text_to_indices(s, tgt_stoi) for s in train_df["German"]]
val_src   = [text_to_indices(s, src_stoi) for s in val_df["English"]]
val_tgt   = [text_to_indices(s, tgt_stoi) for s in val_df["German"]]

train_dataset = TranslationDataset(train_src, train_tgt)
val_dataset = TranslationDataset(val_src, val_tgt)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False,collate_fn=collate_fn)