import logging

from memask.router.intents import Confidence, RoutingResult
from memask.router.llm_classifier import classify_by_llm, is_ollama_available
from memask.router.rules import classify_by_rules

logger = logging.getLogger(__name__)


def route(text: str) -> RoutingResult:
    rules_result = classify_by_rules(text)

    if rules_result.confidence != Confidence.LOW:
        logger.debug(
            "rules classified '%s' as %s (%s)",
            text[:50], rules_result.intent.value, rules_result.confidence.value,
        )
        return rules_result

    if not is_ollama_available():
        logger.debug("ollama unavailable, using rules fallback for '%s'", text[:50])
        return rules_result

    try:
        llm_result = classify_by_llm(text)
    except Exception:
        logger.warning("llm classification failed, falling back to rules")
        return rules_result

    if llm_result is None:
        logger.debug("llm returned None, using rules fallback")
        return rules_result

    logger.debug(
        "llm classified '%s' as %s",
        text[:50], llm_result.intent.value,
    )
    return llm_result
