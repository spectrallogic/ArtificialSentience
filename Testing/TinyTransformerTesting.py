import time, math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

# ---------------------------
# Data
# ---------------------------
text = (
    "An apple fell from the tree today, the apple was nice and red. "
    "The tree was close to another tree. Trees have apples inside of them. "
    "The apple was very red. And the tree fell after the apple fell. "
    "The apple was going to fall from the tree anyways. The tree was big. "
    "And the apple was small."
)

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[i] for i in t.tolist())

data = encode(text)

# ---------------------------
# Hyperparams (tiny)
# ---------------------------
device      = "cuda" if torch.cuda.is_available() else "cpu"
d_model     = 16
n_head      = 1
d_ff        = 32
block_size  = 64
batch_size  = 32
train_steps = 1000
lr          = 3e-3
log_every   = 50      # print sample + details every N steps
bar_width   = 30

# ---------------------------
# Tiny data loader
# ---------------------------
def get_batch():
    ix = torch.randint(low=0, high=max(2, len(data)-block_size-1), size=(batch_size,))
    x  = torch.stack([data[i:i+block_size]      for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1]  for i in ix])
    return x.to(device), y.to(device)

# ---------------------------
# Model
# ---------------------------
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, block_size):
        super().__init__()
        self.key   = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return y

class Block(nn.Module):
    def __init__(self, d_model, block_size, d_ff=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa  = SingleHeadSelfAttention(d_model, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(block_size, d_model)
        self.block     = Block(d_model, block_size, d_ff=d_ff)
        self.ln_f      = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x   = tok + pos
        x   = self.block(x)
        x   = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=160):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

model = TinyTransformer(len(chars), d_model, block_size).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

# ---------------------------
# Text-only logging helpers
# ---------------------------
def ascii_bar(ratio, width=30, fill="#", empty="-"):
    done = int(ratio * width)
    return "[" + fill*done + empty*(width-done) + "]"

def format_topk(probs, k=5):
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        ch = itos[i].replace("\n", "\\n")
        out.append(f"'{ch}': {v:.2f}")
    return " | ".join(out)

@torch.no_grad()
def preview_sample():
    model.eval()
    start = torch.tensor([[stoi[text[0]]]], device=device)
    out = model.generate(start, max_new_tokens=180)[0].cpu()
    model.train()
    return decode(out)

@torch.no_grad()
def preview_distribution(prompt_len=12):
    model.eval()
    seed = text[:prompt_len]
    idx  = encode(seed).unsqueeze(0).to(device)
    logits, _ = model(idx)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    model.train()
    return seed, probs

# ---------------------------
# Train (with console logs)
# ---------------------------
print(f"Device: {device} | Vocab: {len(chars)} | Steps: {train_steps} | LR: {lr}")
loss_ema = None
best_loss = float("inf")
t0 = time.time()

for step in range(1, train_steps+1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()

    # gradient norm (for sanity)
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    grad_norm = math.sqrt(total_norm + 1e-12)

    optim.step()

    # EMA loss + perplexity
    loss_val = loss.item()
    loss_ema = loss_val if loss_ema is None else (0.98 * loss_ema + 0.02 * loss_val)
    ppl = math.exp(min(20, loss_ema))  # clamp to avoid overflow

    # simple progress bar
    ratio = step / train_steps
    bar = ascii_bar(ratio, width=bar_width)
    elapsed = time.time() - t0
    eta = elapsed/ratio - elapsed if ratio > 0 else float("inf")

    line = (
        f"\r{bar} {step:4d}/{train_steps} "
        f"| loss {loss_val:.4f} (ema {loss_ema:.4f}) "
        f"| ppl~{ppl:.2f} | grad {grad_norm:.3f} | lr {lr:.3g} "
        f"| elapsed {elapsed:5.1f}s | eta {eta:5.1f}s"
    )
    sys.stdout.write(line)
    sys.stdout.flush()

    if loss_val < best_loss:
        best_loss = loss_val

    # periodic verbose block
    if step % log_every == 0 or step == 1 or step == train_steps:
        sys.stdout.write("\n")
        # Show a short sample
        sample = preview_sample()
        print("▼ Sample:")
        print(sample)
        # Show next-char distribution for a short prompt
        seed, probs = preview_distribution(prompt_len=min(20, len(text)))
        print(f"▼ Next-char probs after seed: \"{seed}\"")
        print("   " + format_topk(probs, k=5))
        print(f"Best loss so far: {best_loss:.4f}")
        print("-"*80)

print("Training complete.")
