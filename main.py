import pickle
import random
import struct
from collections import Counter

import numpy as np
import tqdm

words = {}
ct = 0
idxs = {}


def word2idx(s: str) -> int:
    global ct
    if s not in words:
        words[s] = ct
        idxs[ct] = s
        ct += 1
    return words[s]


def idx2word(n: int) -> str | None:
    return idxs.get(n)


def sigmoid(n):
    return np.divide(1, 1 + np.exp(-n))


def send_loss(loss):
    loss_file.write(struct.pack("f", loss))
    loss_file.flush()


# load dataset and return it and frequencies
def load_data(file: str) -> tuple[list[str], Counter[str]]:
    corpus = []
    with open(file, "r") as f:
        corpus = f.read().split()

    freq = Counter(corpus)
    return corpus, freq


# subsampling with given params
# threshold - chosen threshold, default 10^-5
# min_count - drop all words with frequency less than, default 5
def subsample(
    corpus: list[str],
    freq: Counter[str],
    threshold: float = 10**-5,
    min_count: int = 5,
) -> list[int]:
    corpus_filtered = []
    n = len(corpus)
    for i in corpus:
        p = 1 - np.sqrt(threshold * n / freq[i])
        if random.random() >= p and freq[i] >= min_count:
            # considerably faster to append than del corpus[j]
            corpus_filtered.append(word2idx(i))

    return corpus_filtered


def train(
    corpus_filtered: list[int],
    epochs: int = 3,
    start_lr: float = 0.025,
    min_lr: float = 0.0001,
    window: int = 5,
    dim: int = 100,
    neg_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for i, j in enumerate(corpus_filtered):
        for offset in range(-window, window + 1):
            if offset != 0 and 0 <= i + offset < len(corpus_filtered):
                pairs.append((j, corpus_filtered[i + offset]))

    V = len(words)
    print(f"vocab size: {V}")
    W_in = np.random.uniform(-0.5, 0.5, size=(V, dim))
    W_out = np.zeros((V, dim))

    pairs_np = np.array(pairs)
    total_steps = len(pairs_np) * epochs
    global_step = 0

    for epoch in range(epochs):
        for wc, wo in tqdm.tqdm(pairs_np, desc=f"epoch {epoch + 1}/{epochs}"):
            rate = max(start_lr * (1 - global_step / total_steps), min_lr)

            uc, uo = W_in[wc], W_out[wo]

            # random negative samples
            samp_idx = np.random.randint(0, V, size=neg_k)
            samp_idx = samp_idx[samp_idx != wc]
            if len(samp_idx) < neg_k:
                samp_idx = np.append(
                    samp_idx,
                    np.random.randint(0, V, size=neg_k - len(samp_idx)),
                )
            samp = W_out[samp_idx]

            # calculate products for gradients
            dot_pos = np.dot(uc, uo)
            sig_pos = sigmoid(dot_pos)
            dot_negs = samp @ uc
            sig_negs = sigmoid(dot_negs)

            g_uc = (sig_pos - 1) * uo + (sig_negs[:, None] * samp).sum(axis=0)
            g_uo = (sig_pos - 1) * uc
            g_uks = sig_negs[:, None] * uc

            W_in[wc] -= rate * g_uc
            W_out[wo] -= rate * g_uo
            W_out[samp_idx] -= rate * g_uks

            # metrics
            if global_step % 50000 == 0:
                obj = -np.log(sig_pos) - np.sum(np.log(1 - sig_negs))
                send_loss(float(obj))

            global_step += 1

    return W_in, W_out


loss_file = open("loss.bin", "ab")

path = "text8"
corpus, freq = load_data(path)
corpus_filt = subsample(corpus, freq)
W_in, W_out = train(
    corpus_filt, epochs=5, start_lr=0.025, window=7, neg_k=15, dim=200
)


np.save("W_in.npy", W_in)
np.save("W_out.npy", W_out)

with open("vocab.pkl", "wb") as f:
    pickle.dump({"words": words, "idxs": idxs, "ct": ct}, f)
