from torch import unsqueeze
import torch
import torch.nn as nn
import torch.nn.functional as f
import math

text = "hello world"
chars = sorted(list(set(text)))
len_vocab = len(chars)

num_char = {i : ch for i , ch in enumerate(chars)}
char_num = {ch:i for i,ch in enumerate(chars)}

def encode(s):
    return [char_num[i] for i in s]

def decoding(s):
    return ''.join([num_char[i] for i in s])

tokens= encode("hello")
token_tensor = torch.tensor(tokens)
token_tensor = token_tensor.unsqueeze(0)

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

block_size = 8
n_embd = 4
token_embedding = nn.Embedding(len_vocab , n_embd)
positional_embedding = nn.Embedding(block_size , n_embd)

tkn_emb = token_embedding(token_tensor)         # (B, T, C)
T = tkn_emb.shape[1]                             # sequence length, NOT the batch dim
pos_emb = positional_embedding(torch.arange(T))  # (T, C): unique position per token

x = tkn_emb + pos_emb
# print(x)

h = head(n_embd=4, head_size=4, block_size=8)
output = h(x)

# print("Output shape:", output.shape)
# print("Output:", output)
        
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


# mha = Multiheadattention(num_heads=2, n_embd=4, head_size=2, block_size=8)
# out = mha(x)   # x is your existing (1, 5, 4) tensor from before
# print(out.shape)

class feedforward(nn.Module):
    def __init__(self , n_embd ,dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd , 4*n_embd) , nn.ReLU() , nn.Linear(4*n_embd , n_embd) , nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
# ffn = feedforward(n_embd=4)
# out = ffn(x)
# print(out.shape , out)

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

# block = Block(n_embd=4, num_heads=2, block_size=8)
# out = block(x)
# print(out.shape , out)

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

model = GPT(len_vocab = len_vocab, n_embd=4, num_heads=2, block_size=10, n_layer=2)
# logits, loss = model(token_tensor)  
# print(logits.shape)  
# print(loss)

text = "hello world"
tokens = encode(text)
data = torch.tensor(tokens)

input = data[:-1].unsqueeze(0)
target = data[1:].unsqueeze(0)

print(input ,"\n" , target)


# logits, loss = model(input, target)
# print(logits.shape)
# print(loss)

Optiomizer = torch.optim.AdamW(model.parameters() , lr= 1e-2)
for step in range(200):
    logits, loss = model(input , target)
    Optiomizer.zero_grad( set_to_none = True)
    loss.backward()
    Optiomizer.step()
    
    if step % 20 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

print("final loss:", loss.item())
