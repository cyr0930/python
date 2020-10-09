import sentencepiece as spm
import random
import os

PATH_DATA = os.path.abspath(f'{__file__}/../../.data')


def get_tokenizer():
    path = f'{PATH_DATA}/tokenizer/m.model'
    return spm.SentencePieceProcessor(model_file=path)


def train_tokenizer():
    path = f'{PATH_DATA}/korean_news_comments/comments.txt'
    if not os.path.isfile(path):
        with open(f'{PATH_DATA}/korean_news_comments/20190101_20200611_v2.txt', encoding='utf-8') as r:
            with open(path, encoding='utf-8', mode='w') as w:
                for line in r:
                    if random.randrange(10) == 0:
                        w.write(line)

    spm.SentencePieceTrainer.Train(
        input=path,
        model_prefix='m',
        vocab_size=30000,
        character_coverage=1.0,
        model_type='bpe',
        user_defined_symbols=[],
    )


if __name__ == '__main__':
    train_tokenizer()
    sp = get_tokenizer()
    tokens = sp.encode('안녕')
    print(tokens)
