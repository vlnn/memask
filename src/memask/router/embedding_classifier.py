import logging

import numpy as np

from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context

logger = logging.getLogger(__name__)

INTENT_EXEMPLARS: dict[Intent, list[str]] = {
    Intent.CAPTURE: [
        "the meeting went well today",
        "kubernetes cluster needs more RAM",
        "talked to alice about the redesign",
        "python 3.12 has nice new features",
        "decided to use postgres instead of mysql",
        "interesting article about distributed systems",
        "the API response time is 200ms",
        "server is under heavy load after the deploy",
        "had to restart the service due to a network stall",
        "the database crashed and recovered on its own",
        "switched to the backup node while primary was down",
        "CI pipeline broke after the OS upgrade",
        "redis memory usage spiked to 90% overnight",
    ],
    Intent.SEARCH: [
        "what did I note about deployment",
        "find my notes about kubernetes",
        "anything about the release plan",
        "notes from the meeting yesterday",
        "what was that thing about databases",
        "deployment notes from last week",
    ],
    Intent.TODO_CREATE: [
        "remind me to buy milk",
        "need to fix the auth bug",
        "should update the documentation",
        "remind about the dentist appointment",
        "have to review the pull request",
        "don't forget to email the client",
    ],
    Intent.TODO_LIST: [
        "show my tasks",
        "what are my todos",
        "list pending items",
        "what do I need to do",
    ],
    Intent.TODO_COMPLETE: [
        "finished buying groceries",
        "done with the code review",
        "mark buy milk as complete",
        "completed the deployment",
    ],
    Intent.TODO_DELETE: [
        "remove buy milk from my list",
        "delete the deployment task",
        "cancel the dentist reminder",
        "remove the meeting todo from my todos",
        "get rid of the grocery task",
    ],
    Intent.TODO_UPDATE: [
        "change the meeting todo to next thursday",
        "update the buy milk task to buy oat milk",
        "reschedule the dentist to friday",
        "rename the deploy task to staging deploy",
        "move the deadline for the report to monday",
    ],
}

SIMILARITY_THRESHOLD = 0.35
_cached_vectors: dict[Intent, np.ndarray] | None = None
_cached_for_service_id: int | None = None


def reset_cache():
    global _cached_vectors, _cached_for_service_id
    _cached_vectors = None
    _cached_for_service_id = None


def classify_by_embedding(text: str, embedding_service) -> RoutingResult | None:
    try:
        vectors = _get_exemplar_vectors(embedding_service)
        query_vec = embedding_service.embed_one(text)
    except Exception:
        logger.debug("embedding classification failed")
        return None

    best_intent = None
    best_score = -1.0

    for intent, exemplar_vecs in vectors.items():
        similarities = exemplar_vecs @ query_vec
        score = float(np.max(similarities))
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_intent is None or best_score < SIMILARITY_THRESHOLD:
        return None

    return RoutingResult(
        intent=best_intent,
        confidence=Confidence.MEDIUM,
        query_context=extract_query_context(text),
        raw_input=text,
        source="embedding",
    )


def _get_exemplar_vectors(embedding_service) -> dict[Intent, np.ndarray]:
    global _cached_vectors, _cached_for_service_id

    service_id = id(embedding_service)
    if _cached_vectors is not None and _cached_for_service_id == service_id:
        return _cached_vectors

    _cached_vectors = {}
    for intent, texts in INTENT_EXEMPLARS.items():
        _cached_vectors[intent] = embedding_service.embed_many(texts)
    _cached_for_service_id = service_id
    return _cached_vectors
