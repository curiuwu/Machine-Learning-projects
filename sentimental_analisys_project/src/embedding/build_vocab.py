PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_word2idx(words: list[str]) -> dict[str, int]:
    word2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word in words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    return word2idx

def build_idx2word(word2idx: dict[str, int]) -> dict[int, str]:
    return {idx: word for word, idx in word2idx.items()}