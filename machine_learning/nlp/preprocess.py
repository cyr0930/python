import random
import os
import hgtk

PATH_DIR = os.path.abspath(f'{__file__}/../../.data')
PATH_ORIGIN = f'{PATH_DIR}/korean_news_comments/20190101_20200611_v2.txt'
PATH_DATA = f'{PATH_DIR}/korean_news_comments/origin.txt'

if __name__ == '__main__':
    if not os.path.isfile(PATH_DATA):
        with open(PATH_ORIGIN, encoding='utf-8') as r:
            with open(PATH_DATA, encoding='utf-8', mode='w') as w:
                for line in r:
                    if random.randrange(10) != 0:
                        continue
                    processed = hgtk.text.decompose(line).replace('ᴥ', '')
                    if len(line) > 256:
                        w.write(processed)
