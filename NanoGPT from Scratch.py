import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 20000
eval_interval = 1000
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # use M1 GPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # if you have cuda
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2

torch.manual_seed(1337)
with open('input.txt','r',encoding = 'utf-8') as f:
    text = f.read()


# here are all the unique characters that occur in this text
	#•	set(text) → removes duplicates, keeps only unique characters from text.
	#•	list(...) → turns the set back into a list (since sets can’t be indexed).
	#•	sorted(...) → sorts the unique characters in order (ASCII/Unicode order).
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi ={ch : i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        
        self.dropout = nn.Dropout(dropout)

def forward(self, x):
    # input: (B, T, C) = batch size, time steps, embedding channels
    # output: (B, T, hs) = batch, time steps, head size

    B, T, C = x.shape  # get shape for convenience

    k = self.key(x)    # (B, T, hs) → project embeddings to key vectors (what each token offers)
    q = self.query(x)  # (B, T, hs) → project embeddings to query vectors (what each token is looking for)

    # compute attention scores (similarity between queries and keys), scale to avoid large values
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)

    # apply causal mask so tokens can't attend to the future
    wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))  # set future positions to -inf

    wei = F.softmax(wei, dim=1)  # turn scores into probabilities (sum to 1 per row)
    wei = self.dropout(wei)      # randomly drop some attention weights for regularization

    v = self.value(x)            # (B, T, hs) → project embeddings to value vectors (information to be passed)
    
    out = wei @ v                # (B, T, T) @ (B, T, hs) → (B, T, hs), weighted sum of values per token

    return out                   # return contextualized token representations

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
   
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 *n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)
    

class Block(nn.Module):
    """ Transformer block: communication (self-attention) followed by computation (feedforward) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # Size of each head by splitting embedding across heads
        self.sa = MultiHeadAttention(n_head, head_size)  # Multi-head self-attention module
        self.ffwd = FeedForward(n_embd)  # Feedforward network after attention
        self.ln1 = nn.LayerNorm(n_embd)  # LayerNorm before attention (pre-norm)
        self.ln2 = nn.LayerNorm(n_embd)  # LayerNorm before feedforward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection: x + self-attention output
        x = x + self.ffwd(self.ln2(x))  # Residual connection: x + feedforward output
        return x  # Final output after attention + feedforward + both residuals
    

class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()  # Initialize the nn.Module

        # Token embedding: converts token IDs into vectors of size n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Positional embedding: gives each position a learnable embedding (since transformers don't have order info)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks: multiple layers of attention + feedforward with residuals and layer norms
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # Final LayerNorm before projecting to vocab size
        self.ln_f = nn.LayerNorm(n_embd)

        # Output linear layer: projects final embeddings to vocab size logits (for prediction)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights (optional but improves training stability)
        self.apply(self._init_weights)


    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean =0.0, std =0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets = None):
        B,T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C =logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the prediction
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax to get the probabilities
            probs = F.softmax(logits, dim=1) #[B,C]
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx
            


model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range (max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval ==0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss{losses['val']:.4f}")

        #SAMPLE A BATCH OF DATA
        xb ,yb = get_batch('train')

        #evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

