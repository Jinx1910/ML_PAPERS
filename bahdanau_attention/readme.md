# Bahdanau Attention Model (PyTorch)

Clean, from-scratch implementation of the **Bahdanau et al. (2014)** neural machine translation model with attention.

### Features
- Bidirectional LSTM encoder
- Additive (Bahdanau) attention
- Full context concatenation (no projection layer — faithful to the paper)
- Proper bidirectional hidden state handling
- No source padding masking (paper used short sentences + bucketing)
