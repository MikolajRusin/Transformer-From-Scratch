import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, block_size: int, n_embed: int, head_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # (B, T, T)

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, block_size: int, num_heads: int, n_embed: int, dropout: float):
        super().__init__()
        head_size = n_embed // num_heads
        self.heads = nn.ModuleList([Head(block_size, n_embed, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads*head_size) ---> (B, T, C)
        out = self.proj(out)  # (B, T, C)
        out = self.dropout(out)  # (B, T, C)
        return out