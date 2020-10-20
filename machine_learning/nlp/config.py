import os

path_dir = os.path.abspath(f'{__file__}/../../.data')
path_data = f'{path_dir}/korean_news_comments/data.txt'

pad_id = 1
eos_id = 2
bptt = 64
batch_size = 32
ntokens = 30000
emsize = 768
nlayers = 12
nhead = 12
dropout = 0.1
initial_weight_scale = 0.02
lr = 1e-4
gladient_clip_val = 1.0
epochs = 100
warmup_steps = 2000
