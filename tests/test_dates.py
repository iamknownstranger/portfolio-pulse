from datetime import date

from common.dates import resolve_start_date

MANUAL_START = date(2024, 1, 15)
# Wednesday, 16 July 2025
END = date(2025, 7, 16)


def test_wtd_starts_on_monday_of_end_week():
    assert resolve_start_date("WTD", MANUAL_START, END) == date(2025, 7, 14)


def test_wtd_on_a_monday_is_that_monday():
    monday = date(2025, 7, 14)
    assert resolve_start_date("WTD", MANUAL_START, monday) == monday


def test_mtd_starts_on_first_of_month():
    assert resolve_start_date("MTD", MANUAL_START, END) == date(2025, 7, 1)


def test_ytd_starts_on_first_of_year():
    assert resolve_start_date("YTD", MANUAL_START, END) == date(2025, 1, 1)


def test_itd_keeps_manual_start():
    assert resolve_start_date("ITD", MANUAL_START, END) == MANUAL_START


def test_unknown_period_keeps_manual_start():
    assert resolve_start_date("whatever", MANUAL_START, END) == MANUAL_START
