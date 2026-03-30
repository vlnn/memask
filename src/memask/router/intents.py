from dataclasses import dataclass, field
from enum import Enum

from memask.router.date_range import DateRange


class Intent(Enum):
    CAPTURE = "capture"
    SEARCH = "search"
    TODO_CREATE = "todo_create"
    TODO_LIST = "todo_list"
    TODO_COMPLETE = "todo_complete"
    APP_COMMAND = "app_command"


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class QueryContext:
    date_hints: list[str] = field(default_factory=list)
    date_range: DateRange | None = None
    topic: str | None = None
    type_filter: str | None = None
    status_filter: str | None = None
    raw_query: str = ""


@dataclass(frozen=True)
class RoutingResult:
    intent: Intent
    confidence: Confidence
    query_context: QueryContext
    raw_input: str
    source: str = "rules"
