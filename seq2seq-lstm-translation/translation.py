from model import Encoder, Decoder, Seq2Seq
from data import SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,tgt_stoi,tgt_pad_idx, src_pad_idx, tgt_pad_idx, src_stoi, tgt_itos
import torch

encoder = Encoder(SRC_VOCAB_SIZE, 256, 0.5, 512, 2)
decoder = Decoder(TGT_VOCAB_SIZE, 256, 0.5, 512, 2)
model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(torch.load("seq2seq_translation_model.pth"))
model.eval()

sentence = "i love you" 
tokens = [src_stoi.get(w, src_stoi["<unk>"]) for w in sentence.lower().split()]
src_tensor = torch.tensor([tokens], device=device)

generated_ids = model.generate(src_tensor, max_len=50)
translation = [tgt_itos[i.item()] for i in generated_ids[0] if i.item() not in [tgt_pad_idx, tgt_stoi["<sos>"], tgt_stoi["<eos>"]]]
print("English:", sentence)
print("German:", " ".join(translation))
