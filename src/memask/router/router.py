import logging
from dataclasses import dataclass

from memask.router.embedding_classifier import classify_by_embedding
from memask.router.intents import Confidence, RoutingResult
from memask.router.rules import classify_by_rules

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterConfig:
    enable_embedding: bool = True


_DEFAULT_CONFIG = RouterConfig()


def route(
    text: str,
    embedding_service=None,
    config: RouterConfig | None = None,
) -> RoutingResult:
    cfg = config or _DEFAULT_CONFIG
    rules_result = classify_by_rules(text)

    if rules_result.confidence == Confidence.HIGH:
        logger.debug(
            "rules classified '%s' as %s (high confidence)",
            text[:50],
            rules_result.intent.value,
        )
        return rules_result

    if not cfg.enable_embedding:
        logger.debug("embedding disabled, using rules for '%s'", text[:50])
        return rules_result

    if embedding_service is None:
        logger.debug("no embedding service, using rules fallback for '%s'", text[:50])
        return rules_result

    try:
        emb_result = classify_by_embedding(text, embedding_service)
    except Exception:
        logger.warning("embedding classification failed, falling back to rules")
        return rules_result

    if emb_result is None:
        logger.debug("embedding returned None, using rules fallback")
        return rules_result

    logger.debug(
        "embedding classified '%s' as %s",
        text[:50],
        emb_result.intent.value,
    )
    return emb_result
