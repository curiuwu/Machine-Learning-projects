import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RNNModel(nn.Module):
    def __init__(self, 
                embedding_matrix, 
                hidden_size=96,
                num_layers=1, 
                num_classes=3,
                dropout=0.3,
                freeze_embeddings=True
                 ):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            padding_idx=0,
            freeze=freeze_embeddings
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout= dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, lengths):
        emdedded = self.embedding_dropout(self.embedding(input_ids))

        packed_embeddings = pack_padded_sequence(
            input=emdedded,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.rnn(packed_embeddings)
        last_hidden = h_n[-1]

        logits = self.classifier(last_hidden)
            
        return logits