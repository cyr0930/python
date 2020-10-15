import sentencepiece as spm
from machine_learning.nlp.config import path_dir, path_data


def get_tokenizer():
    return spm.SentencePieceProcessor(model_file=f'{path_dir}/tokenizer/m.model')


def train_tokenizer():
    spm.SentencePieceTrainer.Train(
        input=path_data,
        model_prefix='m',
        vocab_size=30000,
        character_coverage=1.0,
        model_type='bpe',
        user_defined_symbols=[],
    )


if __name__ == '__main__':
    train_tokenizer()
    sp = get_tokenizer()
    tokens = sp.Encode('안녕')
    print(tokens)
