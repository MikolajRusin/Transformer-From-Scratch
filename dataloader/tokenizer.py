import torch
from dataclasses import dataclass

@dataclass
class CharTokenizer:
    chars: list

    def __post_init__(self):
        self.chars = sorted(self.chars)
        self.vocab_size = len(self.chars)
        self.ch2token = {ch: i for i, ch in enumerate(self.chars)}
        self.token2ch = {i: ch for ch, i in self.ch2token.items()}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.ch2token[ch] for ch in text])

    def decode(self, tokens: torch.Tensor) -> str:
        return ''.join([self.token2ch[token.item()] for token in tokens])