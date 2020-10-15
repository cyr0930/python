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
    data = torch.tensor(data)
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.resize(nbatch, bsz, bptt+1)
    return data.to(device)


def get_batch(data):
    data = data.t()
    source = data[:-1]
    target = data[1:]
    return source, target


tokenizer = get_tokenizer()
data = []
with open(PATH_DATA, encoding='utf-8') as r:
    size = bptt + 1
    for j, line in enumerate(r):
        tokens = tokenizer.encode(line)
        tokens += [eos_id]
        for i in range(0, len(tokens), size):
            buf = tokens[i:i+size]
            if len(buf) > 4:
                if len(buf) < size:
                    buf += [pad_id] * (size - len(buf))
                data.append(buf)
        if j % 10000 == 0:
            print(j)
        if len(data) > 5000:
            break
random.shuffle(data)
train_data = batchify(data, batch_size)

ntokens = len(tokenizer)
emsize = 768
nlayers = 12
nhead = 12
dropout = 0.1
lr = 5e-5
epochs = 100
nbatches = len(train_data)

model = GPT2(emsize, nhead, nlayers, bptt, ntokens, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, epochs * nbatches)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    for i, batch in enumerate(train_data):
        sources, targets = get_batch(batch)
        optimizer.zero_grad()
        with autocast():
            output = model(sources)
            loss = criterion(output.view(-1, ntokens), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        log_interval = 100
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, nbatches, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            print(tokenizer.DecodeIds(sources[:, 0].tolist()))
            print(tokenizer.DecodeIds(output.argmax(-1)[:, 0].tolist()))
            writer.add_scalar('train_loss', cur_loss, (epoch - 1) * nbatches + i)
            total_loss = 0.
            start_time = time.time()


writer = SummaryWriter()
for epoch in range(1, epochs + 1):
    train()
writer.close()
