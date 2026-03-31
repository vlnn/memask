from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from memask.rag.context import assemble_context
from memask.rag.prompts import build_answer_prompt, build_system_message
from memask.rag.reranker import rerank
from memask.search.keyword import SearchResult

if TYPE_CHECKING:
    from memask.context import LLM, Reranker


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[str]
    raw_results: list[SearchResult]
    synthesized: bool


def answer_question(
    query: str,
    results: list[SearchResult],
    *,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
    session_history: list[dict[str, str]] | None = None,
    instructions: list[str] | None = None,
    top_n: int = 5,
    max_context_chars: int = 4000,
) -> AnswerResult:
    reranked = rerank(query, results, reranker, top_n=top_n)
    context = assemble_context(reranked, query, top_n=top_n, max_chars=max_context_chars)

    if not _llm_ready(llm):
        return AnswerResult(
            answer="",
            sources=context.sources,
            raw_results=reranked,
            synthesized=False,
        )

    prompt = build_answer_prompt(query, context, session_history=session_history)
    system = build_system_message(instructions=instructions)
    answer = llm.generate(prompt, system=system)

    return AnswerResult(
        answer=answer,
        sources=context.sources,
        raw_results=reranked,
        synthesized=True,
    )


def _llm_ready(llm: LLM | None) -> bool:
    if llm is None:
        return False
    return llm.is_available()
