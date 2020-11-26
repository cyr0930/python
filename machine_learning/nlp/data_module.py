import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from machine_learning.nlp import config


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        if stage == 'fit':
            self.data = []
            size = config.bptt + 1
            with open(config.path_data, encoding='utf-8') as r:
                for j, line in enumerate(r):
                    tokens = self.tokenizer.encode(line)
                    tokens += [config.eos_id]
                    for i in range(0, len(tokens), size):
                        buf = tokens[i:i + size]
                        if len(buf) > 4:
                            if len(buf) < size:
                                buf += [config.pad_id] * (size - len(buf))
                            self.data.append(buf)
                    if j % 10000 == 0:
                        print(j)
            self.data = TensorDataset(torch.tensor(self.data))

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=config.batch_size, shuffle=True)
