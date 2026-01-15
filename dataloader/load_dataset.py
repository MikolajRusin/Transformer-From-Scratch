import torch
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset

from dataloader.tokenizer import CharTokenizer

@dataclass
class TextLoader(Dataset):
    text: str
    tokenizer: CharTokenizer
    block_size: int
    device: str = 'cpu'

    def __post_init__(self) -> None:
        self.data = self.tokenizer.encode(self.text)

    def __len__(self) -> int:
        return self.data.numel() - self.block_size

    def __getitem__(self, idx: int) -> (torch.tensor, torch.tensor):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        x, y = x.to(self.device), y.to(self.device)
        return x, y