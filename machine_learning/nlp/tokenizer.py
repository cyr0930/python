from tokenizers import ByteLevelBPETokenizer
from machine_learning.nlp.preprocess import PATH_DIR, PATH_DATA

PREFIX = f'{PATH_DIR}/tokenizer'
PATH_VOCAB = f'{PREFIX}/byte-level-vocab-split.json'
PATH_MODEL = f'{PREFIX}/byte-level-merges-split.txt'


def get_tokenizer():
    return ByteLevelBPETokenizer(PATH_VOCAB, PATH_MODEL)


def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[PATH_DATA],
        vocab_size=30000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
    )
    tokenizer.save_model(PREFIX)


if __name__ == '__main__':
    train_tokenizer()
    tokenizer = get_tokenizer()
    encoding = tokenizer.encode('ㅇㅣㅉㅡㅁ ㄷㅚㅁㅕㄴ')
    print(tokenizer.decode(encoding.ids))
