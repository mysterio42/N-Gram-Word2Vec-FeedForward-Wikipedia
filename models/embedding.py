import operator
import re
import string
from functools import partial
from glob import glob

DATA_DIR = '/floyd/input/data/'


class WikipediaEmbedding:

    def __init__(self, vocab_size):
        self.word_freq = {}
        self.word2idx = {}
        self.idx2word = {}
        self.V = vocab_size

    def _enwiki_genexps(self):

        pattern = re.compile(f'[{string.punctuation}]+')
        remove_punctuations = partial(pattern.sub, '')

        return ((sentence for sentence in (remove_punctuations(line).lower().split()
                                           for line in (line for line in open(f))
                                           if line and line[0] not in '[*-|=\{\}')
                 if len(sentence) > 1)
                for f in (file for file in glob(DATA_DIR + 'enwiki*.txt'))
                )

    def _build_word_freq(self):
        for enwiki_sentence_genexp in self._enwiki_genexps():
            for sentence in enwiki_sentence_genexp:
                for word in sentence:
                    self.word_freq[word] = self.word_freq.get(word, 0) + 1

    def _build_word2idx(self):
        top_word_freqs = sorted(self.word_freq.items(), key=operator.itemgetter(1), reverse=True)[:self.V - 1]
        self.word2idx['UNK'] = 0
        self.idx2word[0] = 'UNK'

        for idx, (word, freq) in enumerate(top_word_freqs):
            self.word2idx[word] = idx + 1
            self.idx2word[idx + 1] = word

    def _build_sentence_encoding(self):

        self.encoded_sentences = [
            [self.word2idx[word] if word in self.word2idx else self.word2idx['UNK'] for word in sentence]
            for enwiki_sentence_genexp in self._enwiki_genexps()
            for sentence in enwiki_sentence_genexp
        ]

    def build(self):
        self._build_word_freq()
        self._build_word2idx()
        self._build_sentence_encoding()
