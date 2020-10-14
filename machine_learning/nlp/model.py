import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

INIT_STD = 0.02


class Block(nn.Module):
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
        scaled_std = INIT_STD / math.sqrt(2 * nlayers)
        self.attn.in_proj_weight.data.normal_(std=INIT_STD)
        self.attn.out_proj.weight.data.normal_(std=scaled_std)
        self.linear1.weight.data.normal_(std=INIT_STD)
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
    def __init__(self, embed_dim, nheads, nlayers, num_positions, vocab_size, dropout=0.1):
        super(GPT2, self).__init__()
        self.save_hyperparameters()

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
        self.token_embeddings.weight.data.normal_(std=INIT_STD)
        self.position_embeddings.weight.data.normal_(std=INIT_STD)
        self.head.weight.data.normal_(std=INIT_STD)

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
