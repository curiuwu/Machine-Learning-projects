import torch
from torch.utils.data import Dataset
from src.embedding.build_vocab import PAD_TOKEN, UNK_TOKEN

class ReviewsDataset(Dataset):
    def __init__(self, tokenized_text, labels, word2idx, max_len):
        super().__init__()
        self.tokenized_text = tokenized_text
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def encode_tokens(self, tokens):
        ids = [self.word2idx.get(token, self.word2idx[UNK_TOKEN])
               for token in tokens]
        
        ids = ids[:self.max_len]

        if len(ids) == 0:
            ids = [self.word2idx[UNK_TOKEN]]

        length = len(ids)
        padding_length = self.max_len - length

        ids = ids + [self.word2idx[PAD_TOKEN]] * padding_length

        return ids, length
    
    def __len__(self):
        return len(self.tokenized_text)
    
    def __getitem__(self, idx):
        ids, length = self.encode_tokens(self.tokenized_text[idx])

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
                }