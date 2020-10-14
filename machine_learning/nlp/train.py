import time
import math
import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from machine_learning.nlp.model import GPT2
from machine_learning.nlp.tokenizer import get_tokenizer
from machine_learning.nlp.preprocess import PATH_DATA
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_id = 1
eos_id = 2
bptt = 64
batch_size = 16


def batchify(data, bsz):
    data = torch.tensor([data]).t()
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t()
    data = data.contiguous()
    return data.to(device)


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    mask = target > -1
    return data, target, mask


tokenizer = get_tokenizer()
data = []
with open(PATH_DATA, encoding='utf-8') as r:
    for j, line in enumerate(r):
        tokens = tokenizer.encode(line)
        tokens += [eos_id]
        for i in range(0, len(tokens), bptt):
            buf = tokens[i:i+bptt]
            if len(buf) > 4:
                if len(buf) < bptt:
                    buf += [pad_id] * (bptt - len(buf))
                data.append(buf)
        if j % 10000 == 0:
            print(j)
random.shuffle(data)
train_txt = [y for x in data for y in x]
train_data = batchify(train_txt, batch_size)

ntokens = len(tokenizer)
emsize = 768
nlayers = 12
nhead = 12
dropout = 0.1
lr = 5e-5
epochs = 100
nbatches = len(train_data) // bptt

model = GPT2(emsize, nhead, nlayers, bptt, ntokens, dropout).to(device)
criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, epochs * nbatches)


def train():
    model.train()
    total_loss, total_mask = 0., 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets, mask = get_batch(train_data, i)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += (loss * mask).sum().item()
        total_mask += mask.sum().item()
        log_interval = 100
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / total_mask
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, nbatches, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            print(tokenizer.DecodeIds(data[:, 0].tolist()))
            print(tokenizer.DecodeIds(output.argmax(-1)[:, 0].tolist()))
            writer.add_scalar('loss', cur_loss, (epoch - 1) * nbatches + batch)
            total_loss, total_mask = 0., 0.
            start_time = time.time()


writer = SummaryWriter()
for epoch in range(1, epochs + 1):
    train()
writer.close()
