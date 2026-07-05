# NanoGPT — A From-Scratch GPT Implementation

A minimal, pedagogical re-implementation of the GPT architecture in PyTorch.
Built from first principles, one component at a time, to develop a precise
intuition for how decoder-only transformers actually work — not to compete
with Karpathy's `nanoGPT` on throughput or features.

> **Status:** early stage. Only the tokeniser, embeddings, and a single causal
> self-attention head are implemented so far. The roadmap below describes what
> exists today and what comes next.

---

## Motivation

Reading about transformers and implementing them are very different things.
A surprising amount of the architecture's behavior only becomes clear once
you've written the masking, the scaled dot-product, the residual connections,
and the position embeddings by hand and watched the shapes flow through them.
This project is that exercise: keep the code small enough to hold in your
head, but complete enough to produce a real, trainable language model.

---

## What is implemented today

Everything currently lives in a single file, [`model.py`](model.py):

### 1. Character-level tokeniser

A vocabulary is built from a corpus (`"hello world"` for now), with two
lookup tables mapping characters ↔ integer token IDs.

```python
chars      = sorted(list(set(text)))
num_char   = {i: ch for i, ch in enumerate(chars)}   # id   → char
char_num   = {ch: i for i, ch in enumerate(chars)}   # char → id
```

`encode(s)` converts a string to a list of token IDs; `decoding(s)` is the
inverse. Character-level tokenisation keeps the vocabulary tiny (~10 IDs
for the demo corpus) so the rest of the pipeline is easy to inspect by eye.

### 2. Token and positional embeddings

Two `nn.Embedding` layers — one for tokens, one for positions — summed to
form the input representation fed to the attention layer.

```python
token_embedding      = nn.Embedding(len_vocab, n_embd)   # token   → vector
positional_embedding = nn.Embedding(block_size, n_embd)  # position → vector

x = token_embedding(token_tensor)          # (B, T, C)
x = x + positional_embedding(torch.arange(T))  # add per-position info
```

**Why both?** Token embeddings carry *what* the symbol is; positional
embeddings carry *where* it is. Self-attention is permutation-equivariant
by itself — without positional information, reordering the input tokens
would not change the output beyond the attention weights, which the model
would have no way to interpret.

### 3. Causal self-attention head

The core building block. For an input `x` of shape `(B, T, C)`, it computes
queries, keys, and values via three linear projections, scales the
query–key dot product by `1/sqrt(d_k)`, applies a causal mask using a
pre-registered lower-triangular buffer, softmaxes, and returns the
weighted sum of values.

```python
att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))
att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
att = f.softmax(att, dim=-1)
att = self.dropout(att)
out = att @ v
```

**Why scaled?** For large `d_k`, raw dot products grow in magnitude, which
pushes the softmax into regions with vanishing gradients. Dividing by
`sqrt(d_k)` keeps the variance of the logits at roughly 1 regardless of
head size.

**Why causal masking?** In a language model, token `t` must not attend to
tokens `t+1, t+2, …` from the future. Setting the upper triangle to
`-inf` before softmax gives those positions probability zero after
normalisation. The `register_buffer('tril', ...)` call keeps the mask on
the right device without it being a learnable parameter.

**Why dropout?** Regularisation on the attention weights — discourages
the model from putting all its mass on a single token and helps
generalisation during training.

---

## Running the current code

Requirements: Python 3.8+ with PyTorch installed.

```bash
pip install torch
python model.py
```

The script prints the token embeddings, the positional embeddings, their
cosine similarity (expected to be near zero — the two embedding spaces
are initialised independently), then runs a single attention head over the
embedded `"hello"` sequence and prints the output shape and values.

---

## Roadmap

The architecture is being built up one milestone at a time. Each milestone
keeps the file runnable so the previous behavior can be verified before
moving on.

- [x] **Milestone 1 — Tokeniser & embeddings.** Char-level encode/decode,
      token embeddings, positional embeddings.
- [x] **Milestone 2 — Single causal self-attention head.** Q/K/V
      projections, scaled dot-product, causal mask, dropout.
- [ ] **Milestone 3 — Multi-head attention.** Run `n_head` heads in
      parallel, concatenate along the channel dim, project back to
      `n_embd`.
- [ ] **Milestone 4 — Feed-forward (MLP) block.** Two-layer MLP with a
      GELU non-linearity — the per-token computation path.
- [ ] **Milestone 5 — Residual connections & LayerNorm.** Pre-norm
      transformer block: `x = x + attn(layernorm(x))`, then
      `x = x + mlp(layernorm(x))`.
- [ ] **Milestone 6 — Stack into a full GPT model.** Embedding → N ×
      transformer block → final LayerNorm → LM head (linear back to
      vocabulary).
- [ ] **Milestone 7 — Data loading.** Replace the placeholder corpus with
      a real char-level dataset (tiny-shakespeare is the canonical
      choice), with batched random chunks sliced from the sequence.
- [ ] **Milestone 8 — Training loop.** Cross-entropy loss over the
      shifted targets, AdamW optimiser, periodic loss logging.
- [ ] **Milestone 9 — Generation / sampling.** Temperature-controlled
      autoregressive sampling with top-k truncation, fed back through the
      tokeniser for human-readable output.

---

## Project structure

```
NanoGpt/
├── model.py      # tokeniser, embeddings, and attention head (current)
└── README.md
```

Expected to grow as the roadmap progresses — likely into `model.py`
(full architecture), `data.py` (loading), `train.py` (training loop),
and `generate.py` (inference).

---

## References

- Vaswani et al., *Attention Is All You Need* (2017) — the original
  transformer paper; source of the scaled dot-product attention formula.
- Karpathy, *nanoGPT* — the canonical minimal GPT implementation; this
  project is a slower, more verbose companion to it.
- Karpathy, *Let's build GPT: from scratch, in code, spelled out* — the
  video walkthrough that motivates the milestone ordering above.

---

## License

Personal learning project. No license granted for redistribution at this
time.
