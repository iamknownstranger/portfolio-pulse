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


@pytest.fixture(autouse=True)
def clear_market_cache():
    from common.data import get_price_data, get_top100us_index
    get_price_data.clear()
    get_top100us_index.clear()
    yield
    get_price_data.clear()
    get_top100us_index.clear()


@pytest.mark.parametrize('multi_index', [True, False])
def test_yahoo_adapter_handles_single_ticker_formats_and_inclusive_dates(monkeypatch, multi_index):
    import pandas as pd
    from common import data
    close = pd.DataFrame({'AAPL': [100., 110., 120.]}, index=pd.bdate_range('2025-01-01', periods=3))
    raw = pd.concat({'Close': close}, axis=1) if multi_index else close.rename(columns={'AAPL': 'Close'})
    calls = []

    def download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return raw

    monkeypatch.setattr(data.yf, 'download', download)
    result = data.get_price_data('AAPL', '2025-01-01', '2025-01-03', adjusted=True)
    assert result.columns.tolist() == ['AAPL']
    assert result.iloc[-1, 0] == 120
    assert calls[0][0] == ['AAPL']
    assert calls[0][1]['end'] == '2025-01-04'
    assert calls[0][1]['auto_adjust'] is True


def test_multi_ticker_download_preserves_symbols(monkeypatch):
    import pandas as pd
    from common import data
    close = pd.DataFrame({'AAPL': [100., 110.], 'MSFT': [400., 420.]}, index=pd.bdate_range('2025-01-01', periods=2))
    monkeypatch.setattr(data.yf, 'download', lambda *a, **k: pd.concat({'Close': close}, axis=1))
    pd.testing.assert_frame_equal(data._fetch_from_yfinance(['AAPL', 'MSFT'], '2025-01-01', '2025-01-02'), close, check_freq=False)


@pytest.mark.parametrize('kind', ['missing_ticker', 'stale_window'])
def test_incomplete_local_data_does_not_silently_satisfy_a_request(monkeypatch, kind):
    import pandas as pd
    from common import data
    local = pd.DataFrame({'AAPL': [100., 101.]}, index=pd.to_datetime(['2025-01-02', '2025-01-03']))
    if kind == 'stale_window':
        local['MSFT'] = [200., 201.]
    live = pd.DataFrame({'AAPL': [110., 111.], 'MSFT': [220., 221.]}, index=pd.to_datetime(['2025-01-30', '2025-01-31']))
    monkeypatch.setattr(data, '_fetch_from_duckdb', lambda *a: local)
    monkeypatch.setattr(data, '_fetch_from_yfinance', lambda *a, **k: live)
    pd.testing.assert_frame_equal(data.get_price_data(['AAPL', 'MSFT'], '2025-01-01', '2025-01-31'), live)


def test_adjusted_requests_never_mix_raw_local_prices(monkeypatch):
    import pandas as pd
    from common import data

    def forbidden(*args):
        raise AssertionError('raw store must not be used for adjusted requests')

    monkeypatch.setattr(data, '_fetch_from_duckdb', forbidden)
    live = pd.DataFrame({'AAPL': [100.]})
    monkeypatch.setattr(data, '_fetch_from_yfinance', lambda *a, **k: live)
    pd.testing.assert_frame_equal(data.get_price_data(['AAPL'], '2025-01-01', '2025-01-31', adjusted=True), live)


def test_top100_benchmark_weights_returns_rather_than_share_prices(monkeypatch):
    import pandas as pd
    from common import data
    monkeypatch.setattr(data.pd, 'read_csv', lambda *a: pd.DataFrame({'Symbol': ['A', 'B'], 'marketcap': [300, 200]}))
    prices = pd.DataFrame({'A': [10., 20., 10.], 'B': [500., 500., 1000.]}, index=pd.bdate_range('2025-01-01', periods=3))
    monkeypatch.setattr(data, '_fetch_from_yfinance', lambda *a, **k: prices)
    assert data.get_top100us_index('2025-01-01', '2025-01-03').tolist() == pytest.approx([1, 1.5, 1.875])
