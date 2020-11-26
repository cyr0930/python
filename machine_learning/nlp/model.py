import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW
from machine_learning.nlp import config


class RelativeAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, nheads, dropout=0.1, num_positions=512):
        super().__init__(embed_dim, nheads, dropout=dropout)
        self.num_positions = num_positions
        self.position_embeddings = nn.Embedding(2 * num_positions - 1, embed_dim // nheads)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.bool
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        q = q.contiguous().view(tgt_len, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(tgt_len, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(tgt_len, bsz, self.num_heads, head_dim).permute(1, 2, 0, 3)

        src_len = k.size(2)

        attn_output_weights = torch.matmul(q, k.transpose(-1, -2))
        assert list(attn_output_weights.size()) == [bsz, self.num_heads, tgt_len, src_len]

        position_ids_l = torch.arange(src_len, dtype=torch.long, device=k.device).view(-1, 1)
        position_ids_r = torch.arange(src_len, dtype=torch.long, device=k.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        position_embeddings = self.position_embeddings(distance + self.num_positions - 1)
        position_embeddings = position_embeddings.to(dtype=k.dtype)
        relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", q, position_embeddings)
        relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", k, position_embeddings)
        attn_output_weights = attn_output_weights + relative_position_scores_query + relative_position_scores_key
        attn_output_weights *= scaling

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz, self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, None


class Block(pl.LightningModule):
    def __init__(self, embed_dim, nheads, nlayers, num_positions, dropout):
        super(Block, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = RelativeAttention(embed_dim, nheads, dropout=dropout, num_positions=num_positions)
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
        self.attn.position_embeddings.weight.data.normal_(std=config.initial_weight_scale)

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
    def __init__(self, tokenizer, embed_dim, nheads, nlayers, num_positions, vocab_size,
                 dropout=0.1, tying_weights=True):
        super(GPT2, self).__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id)

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Block(embed_dim, nheads, nlayers, num_positions, dropout))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights(tying_weights)

    def _init_weights(self, tying_weights):
        self.token_embeddings.weight.data.normal_(std=config.initial_weight_scale)
        self.position_embeddings.weight.data.normal_(std=config.initial_weight_scale)
        if tying_weights:
            self.head.weight = self.token_embeddings.weight
        else:
            self.head.weight.data.normal_(std=config.initial_weight_scale)

    def forward(self, x):
        x = x.t()
        h = self.token_embeddings(x.long())
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
