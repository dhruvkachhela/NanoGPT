from torch import unsqueeze
import torch
import torch.nn as nn
import torch.nn.functional as f
import math


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


class head (nn.Module):

    def __init__(self ,n_embd , head_size , block_size , dropout = 0.1 ):
        super().__init__()
        self.Q = nn.Linear( n_embd , head_size , bias = False)
        self.K = nn.Linear( n_embd , head_size , bias = False)
        self.V = nn.Linear( n_embd , head_size , bias = False)
        self.register_buffer('tril' , torch.tril(torch.ones(block_size , block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B , T , C = x.shape
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)

        att = q@k.transpose(-2 ,-1) * (1.0/math.sqrt(k.shape[-1]))
        att = att.masked_fill(self.tril[:T , :T ] ==0 , float('-inf'))
        att = f.softmax(att , dim = -1)
        att = self.dropout(att)
        out = att@v 
        return out
        
class Multiheadattention(nn.Module):
     def __init__(self , n_embd , num_heads , head_size, block_size , dropout = 0.1):
        super().__init__()
        self.head = nn.ModuleList([head(n_embd, head_size , block_size , dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size ,n_embd)
        self.dropout = nn.Dropout(dropout)

     def forward(self, x):
        out = torch.cat([h(x) for h in self.head] , dim = -1)
        out = self.dropout(self.proj(out))
        return out

class feedforward(nn.Module):
    def __init__(self , n_embd ,dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd , 4*n_embd) , nn.ReLU() , nn.Linear(4*n_embd , n_embd) , nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self , n_embd , num_heads , block_size , dropout= 0.1 ):
        super().__init__()
        head_size = n_embd //num_heads
        self.sa = Multiheadattention(n_embd , num_heads , head_size , block_size , dropout)
        self.ffwd = feedforward(n_embd , dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self , x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self , len_vocab , n_embd, num_heads , block_size , n_layer , dropout= 0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(len_vocab , n_embd)
        self.postion_embedding_table = nn.Embedding(block_size , n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd , len_vocab)


    def forward(self , indx , target = None):
        B , T = indx.shape
        tok_embd = self.token_embedding_table(indx)
        pos_embd = self.postion_embedding_table(torch.arange(T))
        x = tok_embd + pos_embd
        x = self.blocks(x)
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


block_size = 32
batch_size = 16

model = GPT(len_vocab=len_vocab, n_embd=64, num_heads=4, block_size=block_size, n_layer=4)
data = torch.tensor(encode(text))

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


for step in range(5000):
    xb, yb = get_batch(train_data, batch_size, block_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

print("final loss:", loss.item())


context = torch.tensor(encode("T")).unsqueeze(0)
out = model.generate(context, max_new_token=200)
print(decode(out[0].tolist()))