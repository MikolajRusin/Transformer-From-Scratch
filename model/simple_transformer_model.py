import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from typing import Union

class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffwd(x)  # (B, T, C)

class TransformerBlock(nn.Module):
    def __init__(self, block_size: int, num_heads: int, n_embed: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(block_size, num_heads, n_embed, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T, C)
        return x

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_transformer_blocks: int, 
        n_embed: int, 
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(block_size, num_heads, n_embed, dropout) for _ in range(n_transformer_blocks)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, targets: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # (batch*block_size, vocab_size)
            targets = targets.view(B*T)   # (batch*block_size, 1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx (B, T)
        for _ in range(max_new_tokens):
            # crop idx to the newest block_size tokens
            idx_cropped = idx[:, -self.position_embedding_table.num_embeddings:]
            # get the predictions
            logits, loss = self(idx_cropped)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply aoftmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append predicted token to the running sequence
            idx = torch.cat((idx, next_idx), dim=-1)
        return idx
