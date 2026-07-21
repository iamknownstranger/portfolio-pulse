"""Pure date helpers (no Streamlit imports so they stay unit-testable)."""

from datetime import timedelta


def resolve_start_date(period, start_date, end_date):
    """Start date implied by a period selection.

    WTD/MTD/YTD derive the start from end_date; ITD (or anything else)
    keeps the manually chosen start_date.
    """
    if period == "WTD":
        return end_date - timedelta(days=end_date.weekday())
    if period == "MTD":
        return end_date.replace(day=1)
    if period == "YTD":
        return end_date.replace(month=1, day=1)
    return start_date
