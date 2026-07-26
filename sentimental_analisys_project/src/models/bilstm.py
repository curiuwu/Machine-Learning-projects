import torch
import torch.nn as nn

from src.models.attention import AttentionLayer
from torch.nn.utils.rnn import (
    pack_padded_sequence, 
    pad_packed_sequence
)




class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
            self,
            embedding_matrix,
            hidden_size,
            num_layers,
            num_classes,
            dropout,
            freeze_embeddings
    ):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = AttentionLayer(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, lengths):
        attention_mask = input_ids != 0

        embedded = self.embedding_dropout(self.embedding(input_ids))

        packed_embeddings = pack_padded_sequence(
            embedded,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.bilstm(packed_embeddings)

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1)
        )

        context_vector, _ = self.attention(
            lstm_output=lstm_output,
            mask=attention_mask
        )

        logits = self.classifier(context_vector)

        return logits