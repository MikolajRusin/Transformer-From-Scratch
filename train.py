import os
import torch
from torch.utils.data import DataLoader
from dataloader.tokenizer import CharTokenizer
from dataloader.load_dataset import TextLoader
from model.simple_transformer_model import TransformerModel
from pathlib import Path

torch.manual_seed(42)

PROJECT_PATH = Path(os.getcwd())
TEXT_PATH = PROJECT_PATH / 'data/input.txt'
SPLIT_RATIO = 0.9

BLOCK_SIZE = 32
BATCH_SIZE = 32
N_TRANSFORMER_BLOCKS = 6
N_EMBED = 128
NUM_HEADS = 8
DROPOUT = 0.2
LEARNING_RATE = 3e-4
N_ITERS = 8000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read text
with open(TEXT_PATH, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))

# Split data into train and valid set
n = int(SPLIT_RATIO*len(text))
train_text = text[:n]
valid_text = text[n:]

# Initialize tokenizer
tokenizer = CharTokenizer(chars)

# Create datasets
train_dataset = TextLoader(train_text, tokenizer=tokenizer, block_size=BLOCK_SIZE, device=DEVICE)
valid_dataset = TextLoader(valid_text, tokenizer=tokenizer, block_size=BLOCK_SIZE, device=DEVICE)

# Initialize dataloaders
train_dataloader = iter(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))
valid_dataloader = iter(DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True))

# Initialize Model
model = TransformerModel(
    block_size=BLOCK_SIZE,
    vocab_size=tokenizer.vocab_size,
    n_transformer_blocks=N_TRANSFORMER_BLOCKS,
    n_embed=N_EMBED,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

n_parameters = sum([p.numel() for p in model.parameters()])

print(f'Train text length: {len(train_text)}')
print(f'Valid text length: {len(valid_text)}')
print(f'Model has been loaded on: {DEVICE}')
print(f'The number of parameters in the model: {n_parameters}')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for dataset in ['train', 'valid']:
        losses = torch.zeros(EVAL_ITERS)
        dataloader = train_dataloader if dataset == 'train' else valid_dataloader
        for k in range(EVAL_ITERS):
            X, Y = next(dataloader)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[dataset] = losses.mean()
    model.train()
    return out

# Training model
for n_iter in range(N_ITERS):
    if n_iter % EVAL_INTERVAL == 0 or n_iter == N_ITERS - 1:
        losses = estimate_loss()
        print(f'step {n_iter}: train loss {losses['train']}, val loss {losses['valid']}')
    
    X_batch, Y_batch = next(train_dataloader)
    
    logits, loss = model(X_batch, Y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Model inference
MAX_NEW_TOKENS = 500
first_token = tokenizer.encode('\n').reshape(1, -1)
print(tokenizer.decode(model.generate(first_token.to(DEVICE), MAX_NEW_TOKENS)[0]))


# BLOCK_SIZE = 8
# BATCH_SIZE = 4
# N_TRANSFORMER_BLOCKS = 4
# N_EMBED = 32
# NUM_HEADS = 4
# DROPOUT = 0.2
# LEARNING_RATE = 1e-3
# N_ITERS = 5000
# EVAL_INTERVAL = 500
# EVAL_ITERS = 200
# step 4999: train loss 2.6712660789489746, val loss 2.67681884765625 (simple bigram model)
# step 4999: train loss 2.499000310897827, val loss 2.5171656608581543 (added positional embedding)
# step 4999: train loss 2.4735045433044434, val loss 2.453296184539795 (added single head attention)
# step 4999: train loss 2.3639376163482666, val loss 2.375704288482666 (added multi head attention)
# step 4999: train loss 2.3533823490142822, val loss 2.384290933609009 (added feed forward)
# step 4999: train loss 2.533794403076172, val loss 2.54364275932312 (added multiple transformer blocks)
# step 4999: train loss 2.3112218379974365, val loss 2.340024471282959 (added residual connection)
# step 4999: train loss 2.2449722290039062, val loss 2.290672540664673 (added layer normalization)
# step 4999: train loss 2.402608871459961, val loss 2.42083740234375 (added dropout)

# ------------ final model ------------
# BLOCK_SIZE = 32
# BATCH_SIZE = 32
# N_TRANSFORMER_BLOCKS = 6
# N_EMBED = 128
# NUM_HEADS = 8
# DROPOUT = 0.2
# LEARNING_RATE = 3e-4
# N_ITERS = 8000
# EVAL_INTERVAL = 500
# EVAL_ITERS = 200

# Train text length: 1003854
# Valid text length: 111540
# Model has been loaded on: cuda
# The number of parameters in the model: 1208385
# step 0: train loss 4.679326057434082, val loss 4.663681507110596
# step 500: train loss 2.3684699535369873, val loss 2.3695497512817383
# step 1000: train loss 2.188693046569824, val loss 2.2127928733825684
# step 1500: train loss 2.071591377258301, val loss 2.122433662414551
# step 2000: train loss 1.9850746393203735, val loss 2.061981201171875
# step 2500: train loss 1.929671287536621, val loss 2.024009943008423
# step 3000: train loss 1.8790452480316162, val loss 1.9764463901519775
# step 3500: train loss 1.8363641500473022, val loss 1.9646694660186768
# step 4000: train loss 1.791213035583496, val loss 1.9315344095230103
# step 4500: train loss 1.7734429836273193, val loss 1.916391134262085
# step 5000: train loss 1.7464669942855835, val loss 1.890831708908081
# step 5500: train loss 1.718196988105774, val loss 1.869357943534851
# step 6000: train loss 1.6982448101043701, val loss 1.860326886177063
# step 6500: train loss 1.6839250326156616, val loss 1.8436845541000366
# step 7000: train loss 1.6626189947128296, val loss 1.8272846937179565
# step 7500: train loss 1.6497737169265747, val loss 1.8289729356765747
# step 7999: train loss 1.6393592357635498, val loss 1.8052252531051636