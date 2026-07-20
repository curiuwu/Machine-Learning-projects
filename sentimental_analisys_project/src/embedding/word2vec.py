from gensim.models import Word2Vec

def train_word2vec(
    tokenized_texts, 
    vector_size: int = 100, 
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 10
) -> Word2Vec:

    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs
    )

    return w2v_model