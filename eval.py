import pickle

import numpy as np


class Word2VecEval:
    def __init__(self, W_in, words, idxs):
        self.words = words
        self.idxs = idxs

        # normalize embeddings once
        self.W = W_in / (np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-10)

    def vector(self, w):
        return self.W[self.words[w]]

    def cosine(self, w1, w2):
        return float(self.vector(w1) @ self.vector(w2))

    def most_similar(self, w, k=10):
        v = self.vector(w)
        sims = self.W @ v

        sims[self.words[w]] = -np.inf
        top = np.argpartition(sims, -k)[-k:]
        top = top[np.argsort(sims[top])[::-1]]

        return [(self.idxs[i], float(sims[i])) for i in top]

    def analogy(self, a, b, c, k=5):
        """
        a - b + c
        """
        v = self.vector(a) - self.vector(b) + self.vector(c)
        v /= np.linalg.norm(v) + 1e-10

        sims = self.W @ v

        for w in (a, b, c):
            sims[self.words[w]] = -np.inf

        top = np.argpartition(sims, -k)[-k:]
        top = top[np.argsort(sims[top])[::-1]]

        return [(self.idxs[i], float(sims[i])) for i in top]


def show(results):
    for w, s in results:
        print(f"{w:<20} {s:.4f}")


W_in = np.load("W_in.npy")
W_out = np.load("W_out.npy")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
    words = vocab["words"]
    idxs = vocab["idxs"]
    ct = vocab["ct"]

w2v = Word2VecEval(W_in, words, idxs)

# tests for
# epochs=5, lr=0.025, dim=200, window=7, neg_k=15

# gives thames 0.4653, england 0.4615
show(w2v.analogy("france", "paris", "london"))
print()
# gives netherlands 0.5915, holland 0.4856
show(w2v.analogy("germany", "berlin", "amsterdam"))
print()
# gives wife 0.4952, married 0.4919 - interesting!
show(w2v.analogy("son", "man", "woman"))
