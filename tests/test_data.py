from datetime import date

import duckdb
import pytest

from common.data import _fetch_from_duckdb


@pytest.fixture
def market_db(tmp_path):
    db_path = str(tmp_path / "market_data.db")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE market_data (date DATE, symbol VARCHAR, close_price DOUBLE)")
    con.executemany(
        "INSERT INTO market_data VALUES (?, ?, ?)",
        [
            (date(2025, 1, 2), "AAPL", 250.0),
            (date(2025, 1, 3), "AAPL", 252.0),
            (date(2025, 1, 2), "MSFT", 430.0),
            (date(2025, 1, 3), "MSFT", 432.0),
            (date(2025, 2, 1), "AAPL", 260.0),  # outside queried range
        ],
    )
    con.close()
    return db_path


class TestFetchFromDuckdb:
    def test_pivots_symbols_to_columns(self, market_db):
        df = _fetch_from_duckdb(["AAPL", "MSFT"], "2025-01-01", "2025-01-31", db_path=market_db)
        assert sorted(df.columns) == ["AAPL", "MSFT"]
        assert len(df) == 2
        assert df.loc[date(2025, 1, 2), "AAPL"] == 250.0

    def test_date_range_is_respected(self, market_db):
        df = _fetch_from_duckdb(["AAPL"], "2025-01-01", "2025-01-31", db_path=market_db)
        assert date(2025, 2, 1) not in df.index

    def test_empty_ticker_list_returns_empty(self, market_db):
        assert _fetch_from_duckdb([], "2025-01-01", "2025-01-31", db_path=market_db).empty

    def test_malicious_symbol_is_treated_as_data(self, market_db):
        # Parameterized query: an injection attempt matches nothing and does no harm
        df = _fetch_from_duckdb(
            ["AAPL'; DROP TABLE market_data; --"],
            "2025-01-01", "2025-01-31", db_path=market_db,
        )
        assert df.empty
        # Table must still exist
        with duckdb.connect(market_db, read_only=True) as con:
            assert con.execute("SELECT COUNT(*) FROM market_data").fetchone()[0] == 5
