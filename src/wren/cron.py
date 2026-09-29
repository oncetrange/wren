"""Cron expressions: the standard five fields, in local time.

    minute hour day-of-month month day-of-week
    */15   9-17 *            *     mon-fri

Fields take `*`, numbers, ranges (`1-5`), lists (`1,3,5`) and steps (`*/15`,
`0-30/10`); months and weekdays also take names (`jan`, `mon`), and Sunday is
0 or 7. Shortcuts: @yearly (@annually), @monthly, @weekly, @daily (@midnight),
@hourly. As in Vixie cron, when both day fields are restricted, a day matches
if either does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

MACROS = {
    "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0", "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@hourly": "0 * * * *",
}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
# (name, low, high, names starting at `low`)
FIELDS = [("minute", 0, 59, None), ("hour", 0, 23, None), ("day of month", 1, 31, None),
          ("month", 1, 12, MONTHS), ("day of week", 0, 7, DAYS)]
# Longest gap between matches worth searching (a Feb 29 schedule can wait 8 years).
SEARCH_YEARS = 9


class CronError(ValueError):
    pass


@dataclass(frozen=True)
class Cron:
    expression: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]  # 0 = Sunday
    any_day: bool  # day of month is *
    any_weekday: bool  # day of week is *

    @classmethod
    def parse(cls, expression: str) -> Cron:
        text = MACROS.get(expression.strip().lower(), expression)
        parts = text.split()
        if len(parts) != 5:
            raise CronError(f"{expression!r}: expected 5 fields (minute hour day month weekday) "
                            "or a shortcut like @daily")
        sets = [_field(p, *spec) for p, spec in zip(parts, FIELDS, strict=True)]
        weekdays = frozenset(d % 7 for d in sets[4])
        return cls(expression.strip(), sets[0], sets[1], sets[2], sets[3], weekdays,
                   any_day=parts[2] == "*", any_weekday=parts[4] == "*")

    def matches(self, t: datetime) -> bool:
        return (t.minute in self.minutes and t.hour in self.hours and t.month in self.months
                and self._day_matches(t))

    def _day_matches(self, t: datetime) -> bool:
        day, weekday = t.day in self.days, (t.weekday() + 1) % 7 in self.weekdays
        if self.any_day or self.any_weekday:
            return day and weekday  # the unrestricted one is always true
        return day or weekday

    def next_after(self, t: datetime) -> datetime | None:
        """The first matching minute strictly after `t`, or None if there's none for years."""
        t = t.replace(second=0, microsecond=0) + timedelta(minutes=1)
        end = t + timedelta(days=366 * SEARCH_YEARS)
        while t < end:
            if t.month not in self.months or not self._day_matches(t):
                t = (t + timedelta(days=1)).replace(hour=0, minute=0)
            elif t.hour not in self.hours:
                t = (t + timedelta(hours=1)).replace(minute=0)
            elif t.minute not in self.minutes:
                t += timedelta(minutes=1)
            else:
                return t
        return None


def _field(text: str, name: str, low: int, high: int, names: list[str] | None) -> frozenset[int]:
    values: set[int] = set()
    for part in text.lower().split(","):
        base, _, step_text = part.partition("/")
        try:
            step = int(step_text) if step_text else 1
        except ValueError:
            raise CronError(f"{name}: bad step {step_text!r}") from None
        if step < 1:
            raise CronError(f"{name}: step must be at least 1")
        if base == "*":
            start, stop = low, high
        elif "-" in base:
            a, b = base.split("-", 1)
            start, stop = _value(a, name, low, high, names), _value(b, name, low, high, names)
            if start > stop:
                raise CronError(f"{name}: range {base!r} runs backwards")
        else:
            start = _value(base, name, low, high, names)
            stop = high if step_text else start  # "5/10" means from 5 on
        values.update(range(start, stop + 1, step))
    return frozenset(values)


def _value(text: str, name: str, low: int, high: int, names: list[str] | None) -> int:
    if names and text in names:
        return names.index(text) + (low if names is MONTHS else 0)
    try:
        value = int(text)
    except ValueError:
        raise CronError(f"{name}: {text!r} is not a number"
                        + (f" or a name ({', '.join(names)})" if names else "")) from None
    if not low <= value <= high:
        raise CronError(f"{name}: {value} is outside {low}-{high}")
    return value


def describe_next(cron: Cron, after: datetime, count: int = 3) -> list[datetime]:
    """The next `count` run times after `after`."""
    times: list[datetime] = []
    t: datetime | None = after
    while len(times) < count and t is not None:
        t = cron.next_after(t)
        if t is not None:
            times.append(t)
    return times
