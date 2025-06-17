import numpy as np
from langchain_core.embeddings import Embeddings


def cosine_similarity_embeddings(
    embedding_a: np.ndarray, embedding_b: np.ndarray
) -> np.ndarray:

    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cos_sim = np.dot(embedding_a, embedding_b) / (norm_a * norm_b)

    if np.isnan(cos_sim):
        print(embedding_a.shape)
        print(embedding_b.shape)
        print(norm_a)
        print(norm_b)
        raise ValueError("Cosine similarity is NaN")

    return cos_sim


def cosine_similarity_text(
    text_a: str, text_b: str, embedding_model: Embeddings
) -> np.ndarray:
    return cosine_similarity_embeddings(
        embedding_a=np.array(embedding_model.embed_query(text_a)),
        embedding_b=np.array(embedding_model.embed_query(text_b)),
    )
