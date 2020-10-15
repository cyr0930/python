import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast
from transformers.optimization import get_cosine_schedule_with_warmup
from machine_learning.nlp import config


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def setup(self, stage):
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


class Block(pl.LightningModule):
    def __init__(self, embed_dim, nheads, nlayers, dropout):
        super(Block, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, nheads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4, bias=False)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim, bias=False)
        self.mlp = nn.Sequential(self.linear1, nn.GELU(), self.linear2)

        self._init_weights(nlayers)

    def _init_weights(self, nlayers):
        scaled_std = config.initial_weight_scale / math.sqrt(2 * nlayers)
        self.attn.in_proj_weight.data.normal_(std=config.initial_weight_scale)
        self.attn.out_proj.weight.data.normal_(std=scaled_std)
        self.linear1.weight.data.normal_(std=config.initial_weight_scale)
        self.linear2.weight.data.normal_(std=scaled_std)
        self.attn.in_proj_bias.data.zero_()
        self.attn.out_proj.bias = None

    def forward(self, x):
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout1(a)
        m = self.mlp(self.ln_2(x))
        x = x + self.dropout2(m)
        return x


class GPT2(pl.LightningModule):
    def __init__(self, tokenizer, embed_dim, nheads, nlayers, num_positions, vocab_size, dropout=0.1):
        super(GPT2, self).__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id)

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Block(embed_dim, nheads, nlayers, dropout))
        self.ln_f = nn.LayerNorm(self.hparams.embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        self.token_embeddings.weight.data.normal_(std=config.initial_weight_scale)
        self.position_embeddings.weight.data.normal_(std=config.initial_weight_scale)
        self.head.weight.data.normal_(std=config.initial_weight_scale)

    def forward(self, x):
        length, batch = x.shape

        h = self.token_embeddings(x.long())

        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)

        h = self.head(h)
        # h = F.linear(h, self.token_embeddings.weight)  # weight tying
        return h

    def configure_optimizers(self):
        total_steps = config.epochs * len(self.train_dataloader())
        optimizers = [torch.optim.Adam(self.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8)]
        schedulers = [{
            'scheduler': get_cosine_schedule_with_warmup(optimizers[0], config.warmup_steps, total_steps),
            'interval': 'step'
        }]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        sources, targets = batch[0][:, :-1].t(), batch[0][:, 1:].t()
        with autocast():
            output = self(sources)
            loss = self.criterion(output.view(-1, config.ntokens), targets.reshape(-1))

        log_interval = 100
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print('\n')
            print(self.tokenizer.DecodeIds(sources[:, 0].tolist()))
            print(self.tokenizer.DecodeIds(output.argmax(-1)[:, 0].tolist()))

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result
