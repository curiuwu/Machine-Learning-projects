import numpy as np
import torch
from src.embedding.build_vocab import UNK_TOKEN

def build_embedding_matrix(w2v_model, word2idx: dict[str, int]) -> torch.Tensor:
    embedding_dim = w2v_model.vector_size
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    embedding_matrix[word2idx[UNK_TOKEN]] = np.random.normal(
        loc=0.0,
        scale=0.1,
        size=(embedding_dim,),
    )

    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]

    return torch.tensor(embedding_matrix, dtype=torch.float32)