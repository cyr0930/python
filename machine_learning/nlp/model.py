import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW
from machine_learning.nlp import config


class Block(pl.LightningModule):
    def __init__(self, embed_dim, nheads, nlayers, dropout):
        super(Block, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, nheads, dropout=dropout)
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
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        self.token_embeddings.weight.data.normal_(std=config.initial_weight_scale)
        self.position_embeddings.weight.data.normal_(std=config.initial_weight_scale)
        # self.head.weight.data.normal_(std=config.initial_weight_scale)
        self.head.weight = self.token_embeddings.weight     # weight tying

    def forward(self, x):
        x = x.t()
        length, batch = x.shape

        h = self.token_embeddings(x.long())

        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)

        h = self.head(h)
        return h.transpose(0, 1)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'ln_']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizers = [AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-8)]
        total_steps = len(self.train_dataloader()) * 10

        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            current_step = min(total_steps, current_step)
            progress = float(current_step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
            return max(0.3, 0.5 * (1.0 + math.cos(math.pi * progress)))

        schedulers = [{
            'scheduler': LambdaLR(optimizers[0], lr_lambda),
            'interval': 'step'
        }]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        sources, targets = batch[0][:, :-1], batch[0][:, 1:]
        output = self(sources)
        loss = self.criterion(output.permute(0, 2, 1), targets)

        log_interval = 100
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print('\n')
            print(self.tokenizer.DecodeIds(sources[0].tolist()))
            print(self.tokenizer.DecodeIds(output.argmax(-1)[0].tolist()))

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result
