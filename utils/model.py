import glob
import json
import os
import random
import string

import numpy as np
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances

WEIGHTS_DIR = 'weights/'
EMBEDDINGS_DIR = 'embeddings/'


def generate_model_name(size=5):
    """
    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def dump_model(word2idx, W, V):
    if not os.path.exists('weights'):
        os.makedirs('weights')

    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    path_json = EMBEDDINGS_DIR + 'word2idx-' + generate_model_name() + '.json'
    path_model = WEIGHTS_DIR + 'w2v_model-' + generate_model_name() + '.npz'

    with open(path_json, 'w') as f:
        json.dump(word2idx, f)
        print(f'word2idx saved at {path_json}')

    np.savez(path_model, W, V)
    print(f'weights saved at {path_model}')


def latest_modified(path_dir):
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(path_dir + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model():
    npz = np.load(latest_modified(WEIGHTS_DIR))
    return npz['arr_0'], npz['arr_1']


def load_embedding():
    with open(latest_modified(EMBEDDINGS_DIR)) as f:
        word2idx = json.load(f)
    return word2idx


def drop_words(sentence, p_drop):
    return [word for word in sentence if np.random.random() < (1 - p_drop[word])]


def word_context(pos, sentence, window_size):
    start = max(0, pos - window_size)
    end_ = min(len(sentence), pos + window_size)

    return np.array([ctx_word_idx for ctx_pos, ctx_word_idx in enumerate(sentence[start:end_], start=start) if
                     ctx_pos != pos])


def neg_sampling_distribution(encoded_sentences, vocab_size):
    word_freq = np.zeros(vocab_size)
    for sentence in encoded_sentences:
        for word in sentence:
            word_freq[word] += 1

    p_neg = word_freq ** 0.75

    p_neg = p_neg / p_neg.sum()

    assert (np.all(p_neg > 0))
    return p_neg
