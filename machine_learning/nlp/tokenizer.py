import sentencepiece as spm
from machine_learning.nlp.preprocess import PATH_DIR, PATH_DATA

DIRECTORY = f'{PATH_DIR}/tokenizer'
NAME = 'byte-level'


def get_tokenizer():
    path = f'{DIRECTORY}/m.model'
    return spm.SentencePieceProcessor(model_file=path)


def train_tokenizer():
    spm.SentencePieceTrainer.Train(
        input=PATH_DATA,
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
