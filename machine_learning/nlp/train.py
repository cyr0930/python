import time
import math
import os
import torch
import torch.nn as nn
from machine_learning.nlp.model import GPT2
from machine_learning.nlp.tokenizer import get_tokenizer

PATH_DATA = os.path.abspath(f'{__file__}/../../.data')
tokenizer = get_tokenizer()
data = []
with open(f'{PATH_DATA}/korean_news_comments/comments.txt', encoding='utf-8') as r:
    for i, line in enumerate(r):
        data += tokenizer.encode(line)
        if i % 10000 == 0:
            print(i)
train_txt = data[:int(len(data) * 0.99)]
val_txt = data[len(train_txt):]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz, shuffle=False):
    data = torch.tensor([data]).t()
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t()
    if shuffle:
        data = data[torch.randperm(data.size()[0])]
    data = data.contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size, True)
val_data = batchify(val_txt, eval_batch_size)

bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


ntokens = len(tokenizer)
emsize = 768
nlayers = 12
nhead = 12
dropout = 0.1
model = GPT2(emsize, nhead, nlayers, bptt, ntokens, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


epochs = 10
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    scheduler.step()
