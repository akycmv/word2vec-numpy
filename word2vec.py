import pickle
import random
import struct
from collections import Counter

import numpy as np
import tqdm


class Word2Vec:
    def __init__(self, path: str):
        self.corpus = []
        self.freq = Counter()

        self.words = {}
        self.idxs = {}
        self.ct = 0

        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None

        self.loss_file = open("loss.bin", "ab")
        self.__load_data(path)

    # index new word or return existing index
    def word2idx(self, s: str) -> int:
        if s not in self.words:
            self.words[s] = self.ct
            self.idxs[self.ct] = s
            self.ct += 1
        return self.words[s]

    # get word by index if it exists
    def idx2word(self, n: int) -> str | None:
        return self.idxs.get(n)

    def __sigmoid(self, n):
        return np.divide(1, 1 + np.exp(-n))

    # save loss datapoints
    def __send_loss(self, loss):
        self.loss_file.write(struct.pack("f", loss))
        self.loss_file.flush()

    # load dataset, space separated
    def __load_data(self, path: str):
        with open(path, "r") as f:
            self.corpus = f.read().split()

        freq = Counter(self.corpus)
        self.freq = freq

    # subsampling with given params
    def subsample(
        self,
        threshold: float = 10**-5,
        min_count: int = 5,
    ) -> list[int]:
        corpus_filtered = []
        n = len(self.corpus)
        for i in self.corpus:
            p = 1 - np.sqrt(threshold * n / self.freq[i])
            if random.random() >= p and self.freq[i] >= min_count:
                # considerably faster to append than del corpus[j]
                corpus_filtered.append(self.word2idx(i))

        return corpus_filtered

    def train_skipgram(
        self,
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

        V = len(self.words)
        W_in = np.random.uniform(-0.5, 0.5, size=(V, dim))
        W_out = np.zeros((V, dim))

        pairs_np = np.array(pairs)
        total_steps = len(pairs_np) * epochs
        global_step = 0

        for epoch in range(epochs):
            for wc, wo in tqdm.tqdm(
                pairs_np, desc=f"epoch {epoch + 1}/{epochs}"
            ):
                rate = max(start_lr * (1 - global_step / total_steps), min_lr)

                uc, uo = W_in[wc], W_out[wo]

                # get negative samples
                samp_idx = np.random.randint(0, V, size=neg_k)
                samp_idx = samp_idx[samp_idx != wc]
                if len(samp_idx) < neg_k:
                    samp_idx = np.append(
                        samp_idx,
                        np.random.randint(0, V, size=neg_k - len(samp_idx)),
                    )
                samp = W_out[samp_idx]

                # forward pass
                # - log sigma(u0^T * uc) - sum log sigma (-uk^T uc)
                dot_pos = np.dot(uc, uo)
                sig_pos = self.__sigmoid(dot_pos)
                dot_negs = samp @ uc
                sig_negs = self.__sigmoid(dot_negs)

                # backward pass
                g_uc = (sig_pos - 1) * uo + (sig_negs[:, None] * samp).sum(
                    axis=0
                )
                g_uo = (sig_pos - 1) * uc
                g_uks = sig_negs[:, None] * uc

                W_in[wc] -= rate * g_uc
                W_out[wo] -= rate * g_uo
                W_out[samp_idx] -= rate * g_uks

                # metrics
                if global_step % 50000 == 0:
                    obj = -np.log(sig_pos) - np.sum(np.log(1 - sig_negs))
                    self.__send_loss(float(obj))

                global_step += 1

        self.loss_file.close()
        self.W_in, self.W_out = W_in, W_out
        return W_in, W_out

    # save run afer finishing
    def save(self):
        np.save("W_in.npy", self.W_in)
        np.save("W_out.npy", self.W_out)

        with open("vocab.pkl", "wb") as f:
            pickle.dump(
                {"words": self.words, "idxs": self.idxs, "ct": self.ct}, f
            )
