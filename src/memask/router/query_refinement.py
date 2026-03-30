import logging
import re

logger = logging.getLogger(__name__)

QUESTION_WORDS = re.compile(
    r"\b(?:what|where|when|how|who|which|why|did|do|does|is|are|was|were|can|could)\b",
    re.I,
)

FILLER_WORDS = re.compile(
    r"\b(?:the|a|an|my|i|me|we|you|he|she|they"
    r"|about|from|that|this|it|its|to|of|in|on|for|with"
    r"|did|do|does|is|are|was|were|can|could|would|should|have|has|had"
    r"|what|where|when|how|who|which|why"
    r"|not|no|so|if|but|and|or)\b",
    re.I,
)

SYSTEM_PROMPT = (
    "You are a search keyword extractor. "
    "Given a natural language query, extract only the core search terms "
    "that should be used to find matching documents. "
    "Strip away question words, pronouns, filler, and conversational phrasing. "
    "Return ONLY the keywords, nothing else. No explanation, no punctuation, no quotes."
)

EXTRACT_PROMPT = "Extract search keywords from: {query}"


def needs_refinement(query: str) -> bool:
    words = query.split()
    if len(words) <= 2:
        return False
    if not QUESTION_WORDS.search(query):
        return False
    content_words = FILLER_WORDS.sub("", query).split()
    if len(content_words) <= 1:
        return False
    return True


def refine_search_query(query: str, llm) -> str:
    try:
        result = llm.generate(
            EXTRACT_PROMPT.format(query=query),
            system=SYSTEM_PROMPT,
        )
        cleaned = _clean_llm_response(result)
        if cleaned:
            logger.debug("refined '%s' -> '%s'", query[:50], cleaned[:50])
            return cleaned
    except Exception:
        logger.debug("query refinement failed, using original")
    return query


def _clean_llm_response(text: str) -> str:
    cleaned = text.strip().strip('"\'`').strip()
    cleaned = re.sub(r"^(?:keywords?:?\s*)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:search\s+(?:for|terms?|keywords?):?\s*)", "", cleaned, flags=re.I)
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[0].strip()
    cleaned = cleaned.strip().rstrip(".,;:!?")
    return cleaned
