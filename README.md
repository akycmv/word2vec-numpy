## word2vec-numpy

Word2Vec implementation using only numpy (additional libraries are only for monitoring progress). The approach used is skip-gram with negative sampling and decaying learning rate; on Macbook M2 with 5 epochs, learning rate of 0.025, embedding vector of length 200, context window of 7 and 15 negative samples the average speed was 35000 it/s, averaging 30 minutes per epoch. Dataset of choice is text8

## Installation
Using [uv](https://github.com/astral-sh/uv):
```
uv sync
uv run main.py
# optionally, to monitor progress: uv run loss.py
# starts flask server with loss chart with smoothing window

# after finishing, eval.py can be used to test results
```

Alternatively, use `pip install -r requirements.txt`
