import torch
import torch.nn as nn




class AttentionLayer(nn.Module):
    def __init__(self, hidden_dimension):
        super().__init__()

        self.attention = nn.Linear(
            hidden_dimension,
            1
        )

    def forward(self, lstm_output, mask):

        scores = self.attention(lstm_output).squeeze(-1)

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        attention_weights_expanded = (
            attention_weights.unsqueeze(-1)
        )

        context_vector = torch.sum(attention_weights_expanded * lstm_output, dim=1)

        return context_vector, attention_weights