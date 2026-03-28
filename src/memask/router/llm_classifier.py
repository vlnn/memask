import logging

import httpx

from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
CLASSIFY_TIMEOUT = 5.0

INTENT_MAP = {member.value: member for member in Intent}

SYSTEM_PROMPT = """You are an intent classifier for a personal memory app.
Classify the user input into exactly one of these categories:
- capture: the user is recording a note, thought, or piece of information
- search: the user is asking a question or looking for something they stored
- todo_create: the user wants to create a task or reminder
- todo_list: the user wants to see their tasks
- todo_complete: the user wants to mark a task as done
- app_command: the user is issuing an application command

Respond with ONLY the category name, nothing else."""


def is_ollama_available() -> bool:
    return _check_ollama()


def _check_ollama() -> bool:
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def classify_by_llm(text: str) -> RoutingResult | None:
    try:
        raw = _call_ollama(text)
    except (ConnectionError, httpx.ConnectError, httpx.TimeoutException, OSError):
        logger.warning("ollama unreachable during classification")
        return None

    intent = INTENT_MAP.get(raw.strip().lower())
    if intent is None:
        logger.warning("unparseable LLM response: %s", raw)
        return RoutingResult(
            intent=Intent.CAPTURE,
            confidence=Confidence.LOW,
            query_context=extract_query_context(text),
            raw_input=text,
            source="llm",
        )

    return RoutingResult(
        intent=intent,
        confidence=Confidence.MEDIUM,
        query_context=extract_query_context(text),
        raw_input=text,
        source="llm",
    )


def _call_ollama(text: str) -> str:
    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": text,
            "stream": False,
        },
        timeout=CLASSIFY_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("response", "")
