from torch import unsqueeze
import torch
import torch.nn as nn
import torch.nn.functional as f
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device : " , device)

with open('flatend.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
len_vocab = len(chars)

num_char = {i: ch for i, ch in enumerate(chars)}
char_num = {ch: i for i, ch in enumerate(chars)}

def encode(s):
    return [char_num[i] for i in s]

def decode(s):
    return ''.join([num_char[i] for i in s])

def precompute_rope_freqs (head_size , block_size , device , base = 10000):
  half_dim = head_size //2
  freqns = 1.0 / (base** ( torch.arange(0 ,half_dim , device=device).float() / half_dim))
  t = torch.arange(block_size , device= device).float()
  freqns = torch.outer(t , freqns)
  sin = torch.sin(freqns)
  cos = torch.cos(freqns)
  return sin , cos

def apply_rope(x, cos, sin):
    B, T, hs = x.shape
    x1 = x[..., :hs // 2]
    x2 = x[..., hs // 2:]
    cos = cos[:T].unsqueeze(0)
    sin = sin[:T].unsqueeze(0)
    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos
    return torch.cat([rotated1, rotated2], dim=-1)


class head (nn.Module):

    def __init__(self ,n_embd , head_size , block_size , dropout = 0.2 ):
        super().__init__()
        self.Q = nn.Linear( n_embd , head_size , bias = False)
        self.K = nn.Linear( n_embd , head_size , bias = False)
        self.V = nn.Linear( n_embd , head_size , bias = False)
        self.register_buffer('tril' , torch.tril(torch.ones(block_size , block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x , rop_cos , rop_sin):
        B , T , C = x.shape
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        q = apply_rope( q , rop_cos , rop_sin)
        k = apply_rope (k , rop_cos , rop_sin)
        att = q@k.transpose(-2 ,-1) * (1.0/math.sqrt(k.shape[-1]))
        att = att.masked_fill(self.tril[:T , :T ] ==0 , float('-inf'))
        att = f.softmax(att , dim = -1)
        att = self.dropout(att)
        out = att@v
        return out

class Multiheadattention(nn.Module):
     def __init__(self , n_embd , num_heads , head_size, block_size , dropout = 0.2):
        super().__init__()
        self.head = nn.ModuleList([head(n_embd, head_size , block_size , dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size ,n_embd)
        self.dropout = nn.Dropout(dropout)

     def forward(self, x , rop_cos , rop_sin):
        out = torch.cat([h(x , rop_cos , rop_sin) for h in self.head] , dim = -1)
        out = self.dropout(self.proj(out))
        return out

class feedforward(nn.Module):
    def __init__(self , n_embd ,dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd , 4*n_embd) , nn.ReLU() , nn.Linear(4*n_embd , n_embd) , nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self , n_embd , num_heads , block_size , dropout= 0.2 ):
        super().__init__()
        head_size = n_embd //num_heads
        self.sa = Multiheadattention(n_embd , num_heads , head_size , block_size , dropout)
        self.ffwd = SWIGlu(n_embd , dropout)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self , x , rop_cos , rop_sin):
        x = x + self.sa(self.ln1(x) , rop_cos , rop_sin)
        x = x + self.ffwd(self.ln2(x))
        return x

class RMSNorm(nn.Module):
  def __init__(self , dim , eps = 1e-6):
    super().__init__()
    self.eps = eps
    self.weights = nn.Parameter(torch.ones(dim))

  def forward(self , x):
      rms = x.pow(2).mean( dim = -1 , keepdim = True).add(self.eps).sqrt()
      return x/rms *self.weights

class SWIGlu(nn.Module):
  def __init__(self , n_embd , dropout=0.2):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.w1 = nn.Linear(n_embd , 4*n_embd , bias = False)
    self.w3 = nn.Linear(n_embd , 4*n_embd , bias = False)
    self.w2 = nn.Linear (4*n_embd , n_embd , bias = False)

  def forward(self , x):
    return self.dropout(self.w2(f.silu(self.w1(x)) * self.w3(x)))

class GPT(nn.Module):
    def __init__(self , len_vocab , n_embd, num_heads , block_size , n_layer , dropout= 0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(len_vocab , n_embd)
        
        head_size = n_embd // num_heads
        rop_cos, rop_sin = precompute_rope_freqs(head_size, block_size, device)
        self.register_buffer('rop_cos', rop_cos)
        self.register_buffer('rop_sin', rop_sin)

        self.blocks = nn.ModuleList([Block(n_embd, num_heads, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd , len_vocab)


    def forward(self , indx , target = None):
        B , T = indx.shape
        x = self.token_embedding_table(indx)

        for block in self.blocks:
          x = block(x , self.rop_cos , self.rop_sin)
      
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target == None:
            loss = None
        else:
            B , T , C = logits.shape
            logits = logits.view(B*T ,C )
            target = target.view(B*T)
            loss = f.cross_entropy(logits , target)
        return logits, loss

    def generate(self , indx , max_new_token):
        for _ in range(max_new_token):
            indx_cond = indx[:, -self.block_size:]
            logits, _ = self(indx_cond)
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim=-1)
            indx_next = torch.multinomial(probs, num_samples=1)
            indx = torch.cat((indx, indx_next), dim=1)
        return indx

def get_batch(data , batch_size , block_size):
    ix = torch.randint(len(data)-block_size, (batch_size , ) )
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+1+block_size] for i in ix])
    return x , y

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = get_batch(data, batch_size, block_size)
        _, loss = model(xb, yb)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


block_size = 64
batch_size = 32

model = GPT(len_vocab=len_vocab, n_embd=256, num_heads=8, block_size=block_size, n_layer=6)
model = model.to(device)
data = torch.tensor(encode(text)).to(device)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
best_val_loss = float('inf')

for step in range(2000):
    xb, yb = get_batch(train_data, batch_size, block_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        train_loss = estimate_loss(model, train_data, block_size, batch_size)
        val_loss = estimate_loss(model, val_data, block_size, batch_size)
        print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("new best val loss, saved checkpoint")

print("final loss:", loss.item())
print("best val loss:", best_val_loss)

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

context = torch.tensor(encode("T")).unsqueeze(0).to(device)
out = model.generate(context, max_new_token=300)
print(decode(out[0].tolist()))