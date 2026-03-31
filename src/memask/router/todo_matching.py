import numpy as np

from memask.models.item import Item


def find_todos(query: str, todos: list[Item]) -> list[Item]:
    query_lower = query.lower()
    return [t for t in todos if query_lower in t.content.lower()]


def find_todos_semantic(
    query: str,
    todos: list[Item],
    embedder,
    threshold: float = 0.5,
) -> list[Item]:
    if not todos or embedder is None:
        return []

    query_vec = np.asarray(embedder.embed_one(query), dtype=np.float32)
    todo_texts = [t.content for t in todos]
    todo_vecs = np.asarray(embedder.embed_many(todo_texts), dtype=np.float32)

    scores = _cosine_similarities(query_vec, todo_vecs)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        return [todos[best_idx]]
    return []


def _cosine_similarities(query_vec, candidate_vecs):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-9
    normalized = candidate_vecs / norms
    return normalized @ query_norm
