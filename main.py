import word2vec

path = "text8"
w2v = word2vec.Word2Vec(path)
corpus = w2v.subsample(threshold=10**-5, min_count=5)

W_in, W_out = w2v.train_skipgram(
    corpus, epochs=5, start_lr=0.025, window=7, neg_k=15, dim=200
)

# save results
w2v.save()
