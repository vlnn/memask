import re
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime
    label: str


OFFSET_RE = re.compile(r"^([+-])(\d+)([dwmy])$", re.I)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LAST_WEEKDAY_RE = re.compile(
    r"^last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",
    re.I,
)

WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

UNIT_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def resolve(expression: str, now: datetime | None = None) -> DateRange | None:
    now = now or datetime.now()
    expr = expression.strip().lower()

    m = OFFSET_RE.match(expr)
    if m:
        return _resolve_offset(m.group(1), int(m.group(2)), m.group(3), now)

    handler = _NAMED_HANDLERS.get(expr)
    if handler:
        return handler(now)

    m = LAST_WEEKDAY_RE.match(expr)
    if m:
        return _resolve_last_weekday(m.group(1).lower(), now)

    if ISO_DATE_RE.match(expr):
        try:
            d = datetime.fromisoformat(expr)
            return _day_range(d, expr)
        except ValueError:
            return None

    return None


def _resolve_offset(sign, amount, unit, now):
    delta = timedelta(days=amount * UNIT_DAYS[unit.lower()])

    if sign == "-":
        return DateRange(
            start=_start_of_day(now - delta),
            end=_end_of_day(now),
            label=f"last {amount}{unit.lower()}",
        )
    return DateRange(
        start=_start_of_day(now),
        end=_end_of_day(now + delta),
        label=f"next {amount}{unit.lower()}",
    )


def _resolve_today(now):
    return _day_range(now, "today")


def _resolve_yesterday(now):
    return _day_range(now - timedelta(days=1), "yesterday")


def _resolve_tomorrow(now):
    return _day_range(now + timedelta(days=1), "tomorrow")


def _resolve_this_week(now):
    start = now - timedelta(days=now.weekday())
    return DateRange(
        start=_start_of_day(start),
        end=_end_of_day(now),
        label="this week",
    )


def _resolve_last_week(now):
    return DateRange(
        start=_start_of_day(now - timedelta(days=7)),
        end=_end_of_day(now),
        label="last week",
    )


def _resolve_this_month(now):
    start = now.replace(day=1)
    return DateRange(
        start=_start_of_day(start),
        end=_end_of_day(now),
        label="this month",
    )


def _resolve_last_month(now):
    return DateRange(
        start=_start_of_day(now - timedelta(days=30)),
        end=_end_of_day(now),
        label="last month",
    )


def _resolve_this_year(now):
    start = now.replace(month=1, day=1)
    return DateRange(
        start=_start_of_day(start),
        end=_end_of_day(now),
        label="this year",
    )


def _resolve_last_year(now):
    return DateRange(
        start=_start_of_day(now - timedelta(days=365)),
        end=_end_of_day(now),
        label="last year",
    )


def _resolve_last_weekday(day_name, now):
    target = WEEKDAY_INDEX[day_name]
    days_back = (now.weekday() - target) % 7
    if days_back == 0:
        days_back = 7
    dt = now - timedelta(days=days_back)
    return _day_range(dt, f"last {day_name}")


_NAMED_HANDLERS = {
    "today": _resolve_today,
    "yesterday": _resolve_yesterday,
    "tomorrow": _resolve_tomorrow,
    "this week": _resolve_this_week,
    "last week": _resolve_last_week,
    "this month": _resolve_this_month,
    "last month": _resolve_last_month,
    "this year": _resolve_this_year,
    "last year": _resolve_last_year,
}


def _day_range(dt, label):
    return DateRange(start=_start_of_day(dt), end=_end_of_day(dt), label=label)


def _start_of_day(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt):
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
