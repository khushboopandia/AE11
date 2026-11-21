"""Stage 3 and 4: embed the chunks, store them, and search them.

YOUR CODE GOES HERE. `Hit` and the OpenAI client are written for you, because
plumbing is not the lesson. `embed_texts`, `cosine_similarity`, `build_index`
and `search` are the lesson.
"""

from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from collections import Counter

EMBEDDING_MODEL = "text-embedding-3-small"
AGE_BANDS=["Age band: all ages", "Age band: under 4 years", "Age band: 4 to 5 years", "Age band: 6 to 11 years", "Age band: 12 to 17 years"]

_cached: OpenAI | None = None


def client() -> OpenAI:
    """The OpenAI client, built on first use and reused after that.

    It is built lazily, not at import time, so that .env has already been
    loaded by the time it looks for OPENAI_API_KEY.
    """
    global _cached
    if _cached is None:
        _cached = OpenAI()
    return _cached


@dataclass
class Hit:
    """One retrieved chunk, and everything you need to cite it.

    `chunk_id` is what makes an answer traceable: when an answer is wrong in
    Week 4, this is how you find out which chunk caused it. Keep it.
    """

    chunk_id: int
    text: str
    score: float

    def citation(self) -> str:
        """A short, human-readable label for this chunk. Printed under answers."""
        preview = " ".join(self.text.split())[:60]
        return f"[chunk {self.chunk_id}] \n\n {self.text}... \n\n (score {self.score:.3f})"


def allowed_age_bands(age_text):
    """'2 years' -> ['Age band: all ages', 'Age band: under 4 years']"""
    years = float(age_text.split()[0])      # '0.9 years' -> 0.9

    if years < 4:
        band = "under 4 years"
    elif years < 6:
        band = "4 to 5 years"
    elif years < 12:
        band = "6 to 11 years"
    elif years < 18:
        band = "12 to 17 years"
    else:
        band = None                          # 18+: all ages only

    allowed = ["Age band: all ages"]
    if band:
        allowed.append(f"Age band: {band}")
    return allowed

def embed_texts(texts: list[str], input_type: str = "document") -> np.ndarray:
    """Turn a list of strings into a matrix of embedding vectors.

    Args:
        texts: the strings to embed. One API call can take many at once —
            do NOT loop and call once per chunk.

    Returns:
        An array of shape (len(texts), 1536), one row per input string,
        in the same order as `texts`.

    Hint: `client().embeddings.create(input=texts, model=EMBEDDING_MODEL)`.
    The response has a `.data` list; each item has an `.embedding`. Wrap the
    result in np.array(). Print `response.usage` once — knowing what a corpus
    costs to embed is the number that decides architecture at scale.
    """

    if not texts:
        return np.empty((0, 1024), dtype=np.float32)
    
    result = client().embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    embeddings = np.array([item.embedding for item in result.data], dtype=np.float32)
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How similar are two vectors? 1.0 is identical, 0.0 is unrelated.

    Hint: the dot product, divided by the product of the two norms.
    NumPy has `np.dot` and `np.linalg.norm`.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_index(chunks: list[str]) -> tuple[np.ndarray, list[str]]:
    """Embed every chunk once, and return the matrix to search against.

    Args:
        chunks: from corpus.split_corpus().

    Returns:
        The embedding matrix. Row i corresponds to chunks[i] — that
        correspondence is the only thing holding your citations together,
        so do not sort or filter one without the other.

    Hint: this is mostly a call to embed_texts(). Notice that it happens once
    per process, and that a fresh process has no memory of the last one.
    """

    chunk_metadata=[]
    age_band_count = Counter()
    for chunk in chunks:
        found = False
        for age_band in AGE_BANDS:
            if age_band in chunk.split("\n")[0]:
                found = True
                chunk_metadata.append({"age_band": age_band})
                age_band_count[age_band] += 1
                break
        if not found:
            age_band_count["not_found"] += 1
            chunk_metadata.append({"age_band": "not_found"})
    print(age_band_count)
    return embed_texts(chunks), chunk_metadata

def search(query: str, age:str, chunks: list[str], chunk_metadata: list[str], matrix: np.ndarray, k: int = 3) -> list[Hit]:
    """Find the k chunks most similar to the query.

    Args:
        query: the user's question, as typed.
        chunks: the chunk texts.
        matrix: the embedding matrix from build_index().
        k: how many hits to return.

    Returns:
        Up to k Hits, most similar first.

    Hints:
      - Embed the query with the SAME model you built the index with. A query
        embedded by a different model is not comparable, and the failure is
        silent — you get plausible-looking nonsense, not an error.
      - Score the query against every row, then sort.
      - Read the signature again: it says "up to k". Nothing here forces you to
        return exactly k. That freedom is the whole of Step 4's threshold option.
    """
    if not query:
        raise ValueError("The query to search against is empty.")
    if not chunks or matrix.size == 0:
        raise ValueError("The chunks and/or matrix is empty.")
    if k <= 0:
        raise ValueError("Retrieve one or more documents.")

    query_embedding = embed_texts([f"ChildAge: {age} {query}"], "query")[0] 
    scores = []   
    for chunk_embedding in matrix:
        scores.append(cosine_similarity(query_embedding, chunk_embedding))
    
    top_k = min(k, len(scores))
    hits = []
    top_indices = np.argsort(scores)[::-1][:top_k]
    for top_index in top_indices:
        hit  = Hit(int(top_index), chunks[top_index], scores[top_index])
        hits.append(hit)

    return hits

def search_intake_agent(query: str, age:str, chunks: list[str], chunk_metadata: list[str], matrix: np.ndarray, k: int = 3) -> list[Hit]:
    """Find the k chunks most similar to the query.

    Args:
        query: the user's question, as typed.
        chunks: the chunk texts.
        matrix: the embedding matrix from build_index().
        k: how many hits to return.

    Returns:
        Up to k Hits, most similar first.

    Hints:
      - Embed the query with the SAME model you built the index with. A query
        embedded by a different model is not comparable, and the failure is
        silent — you get plausible-looking nonsense, not an error.
      - Score the query against every row, then sort.
      - Read the signature again: it says "up to k". Nothing here forces you to
        return exactly k. That freedom is the whole of Step 4's threshold option.
    """
    if not query:
        raise ValueError("The query to search against is empty.")
    if not chunks or matrix.size == 0:
        raise ValueError("The chunks and/or matrix is empty.")
    if k <= 0:
        raise ValueError("Retrieve one or more documents.")

    query_embedding = embed_texts([f"ChildAge: {age} {query}"], "query")[0]

    allowed = allowed_age_bands(age)
    mask = np.array([b["age_band"] in allowed for b in chunk_metadata])
    idx = np.where(mask)[0]
    assert len(idx) > 0
    assert mask.shape[0] == matrix.shape[0]
    filtered_matrix = matrix[idx]
    scores = []   
    for chunk_embedding in filtered_matrix:
        scores.append(cosine_similarity(query_embedding, chunk_embedding))
    
    top_k = min(k, len(scores))
    hits = []
    top_indices = np.argsort(scores)[::-1][:top_k]
    for top_index in top_indices:
        chunk_id = int(idx[top_index])
        hit = Hit(chunk_id, chunks[chunk_id], scores[top_index])
        hits.append(hit)
    print("DEBUG: " , len(idx), len(top_indices), len(hits))
    return hits

