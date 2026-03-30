from datetime import datetime, timedelta

import pytest

from memask.router.date_range import DateRange, resolve


NOW = datetime(2026, 3, 30, 14, 30, 0)


class TestOffsetResolution:
    @pytest.mark.parametrize("expr,expected_label", [
        ("-1d", "last 1d"),
        ("-3d", "last 3d"),
        ("-1w", "last 1w"),
        ("-2w", "last 2w"),
        ("-1m", "last 1m"),
        ("-3m", "last 3m"),
        ("-1y", "last 1y"),
    ])
    def test_negative_offset_produces_range(self, expr, expected_label):
        result = resolve(expr, now=NOW)
        assert result is not None, f"'{expr}' should resolve to a DateRange"
        assert result.label == expected_label, (
            f"'{expr}' should have label '{expected_label}'"
        )
        assert result.start < result.end, "start should be before end"

    @pytest.mark.parametrize("expr,expected_label", [
        ("+1d", "next 1d"),
        ("+3d", "next 3d"),
        ("+1w", "next 1w"),
    ])
    def test_positive_offset_produces_future_range(self, expr, expected_label):
        result = resolve(expr, now=NOW)
        assert result is not None, f"'{expr}' should resolve to a DateRange"
        assert result.label == expected_label, (
            f"'{expr}' should have label '{expected_label}'"
        )
        assert result.end > NOW, "positive offset end should be in the future"

    def test_negative_1d_covers_yesterday_to_now(self):
        result = resolve("-1d", now=NOW)
        yesterday_start = datetime(2026, 3, 29, 0, 0, 0)
        assert result.start == yesterday_start, "start should be yesterday midnight"
        assert result.end.date() == NOW.date(), "end should be today"

    def test_negative_1w_covers_7_days(self):
        result = resolve("-1w", now=NOW)
        expected_start = datetime(2026, 3, 23, 0, 0, 0)
        assert result.start == expected_start, "start should be 7 days ago midnight"

    def test_case_insensitive(self):
        result = resolve("-1W", now=NOW)
        assert result is not None, "offset should be case insensitive"
        assert result.label == "last 1w", "label should use lowercase unit"


class TestNamedPeriods:
    def test_today(self):
        result = resolve("today", now=NOW)
        assert result.label == "today", "should resolve to 'today'"
        assert result.start == datetime(2026, 3, 30, 0, 0, 0), (
            "should start at midnight today"
        )
        assert result.end.date() == NOW.date(), "should end today"

    def test_yesterday(self):
        result = resolve("yesterday", now=NOW)
        assert result.label == "yesterday", "should resolve to 'yesterday'"
        assert result.start == datetime(2026, 3, 29, 0, 0, 0), (
            "should start at midnight yesterday"
        )
        assert result.end.date() == datetime(2026, 3, 29).date(), (
            "should end yesterday"
        )

    def test_tomorrow(self):
        result = resolve("tomorrow", now=NOW)
        assert result.label == "tomorrow", "should resolve to 'tomorrow'"
        assert result.start.date() == datetime(2026, 3, 31).date(), (
            "should start tomorrow"
        )

    def test_this_week_starts_at_monday(self):
        result = resolve("this week", now=NOW)
        assert result.label == "this week", "should resolve to 'this week'"
        assert result.start.weekday() == 0, "should start on Monday"
        assert result.start <= NOW, "start should be before or at now"

    def test_last_week_is_sliding_7_days(self):
        result = resolve("last week", now=NOW)
        expected_start = datetime(2026, 3, 23, 0, 0, 0)
        assert result.start == expected_start, "last week should be sliding 7 days back"

    def test_this_month_starts_at_first(self):
        result = resolve("this month", now=NOW)
        assert result.start.day == 1, "this month should start on the 1st"
        assert result.start.month == NOW.month, "should be current month"

    def test_last_month_is_sliding_30_days(self):
        result = resolve("last month", now=NOW)
        expected_start = datetime(2026, 2, 28, 0, 0, 0)
        assert result.start == expected_start, "last month should be sliding 30 days"

    def test_this_year_starts_jan_1(self):
        result = resolve("this year", now=NOW)
        assert result.start.month == 1, "this year should start in January"
        assert result.start.day == 1, "this year should start on the 1st"

    def test_last_year_is_sliding_365_days(self):
        result = resolve("last year", now=NOW)
        delta = NOW - result.start
        assert 364 <= delta.days <= 366, "last year should be ~365 days back"


class TestLastWeekday:
    @pytest.mark.parametrize("day_name,expected_weekday", [
        ("monday", 0),
        ("tuesday", 1),
        ("wednesday", 2),
        ("thursday", 3),
        ("friday", 4),
        ("saturday", 5),
        ("sunday", 6),
    ])
    def test_resolves_to_correct_weekday(self, day_name, expected_weekday):
        result = resolve(f"last {day_name}", now=NOW)
        assert result is not None, f"'last {day_name}' should resolve"
        assert result.start.weekday() == expected_weekday, (
            f"'last {day_name}' should land on {day_name}"
        )

    def test_last_same_weekday_goes_back_7_days(self):
        now = datetime(2026, 3, 30, 12, 0, 0)
        result = resolve("last monday", now=now)
        assert result.start == datetime(2026, 3, 23, 0, 0, 0), (
            "last monday on a monday should go back 7 days"
        )

    def test_last_weekday_is_single_day_range(self):
        result = resolve("last friday", now=NOW)
        assert result.start.date() == result.end.date(), (
            "last <weekday> should be a single-day range"
        )


class TestIsoDate:
    def test_resolves_valid_iso_date(self):
        result = resolve("2026-03-15", now=NOW)
        assert result is not None, "should resolve ISO date"
        assert result.start == datetime(2026, 3, 15, 0, 0, 0), (
            "should start at midnight of given date"
        )
        assert result.label == "2026-03-15", "label should be the ISO date"

    def test_single_day_range(self):
        result = resolve("2026-01-01", now=NOW)
        assert result.start.date() == result.end.date(), (
            "ISO date should be a single-day range"
        )


class TestUnresolvable:
    @pytest.mark.parametrize("expr", [
        "banana",
        "next quarter",
        "2 days ago",
        "",
        "   ",
        "1d",
        "w",
    ])
    def test_returns_none(self, expr):
        result = resolve(expr, now=NOW)
        assert result is None, f"'{expr}' should not resolve to a DateRange"


class TestDayBoundaries:
    def test_start_of_day(self):
        result = resolve("today", now=NOW)
        assert result.start.hour == 0, "start should be at hour 0"
        assert result.start.minute == 0, "start should be at minute 0"
        assert result.start.second == 0, "start should be at second 0"

    def test_end_of_day(self):
        result = resolve("today", now=NOW)
        assert result.end.hour == 23, "end should be at hour 23"
        assert result.end.minute == 59, "end should be at minute 59"
        assert result.end.second == 59, "end should be at second 59"
