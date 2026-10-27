from datetime import date

import duckdb
import pytest

from common.indices import archive_bounds, read_membership


@pytest.fixture
def archive(tmp_path):
    path = str(tmp_path / 'archive.db')
    with duckdb.connect(path) as con:
        con.execute('CREATE TABLE market_cap_data (symbol VARCHAR, date DATE, market_cap DOUBLE)')
        con.executemany('INSERT INTO market_cap_data VALUES (?, ?, ?)', [
            ('A', '2025-01-02', 300), ('B', '2025-01-02', 200), ('C', '2025-01-02', 100),
            ('A', '2025-01-03', 300), ('B', '2025-01-03', 200), ('C', '2025-01-03', 400),
        ])
    return path


def test_point_in_time_membership_and_changes(archive):
    members, snapshot, removed = read_membership('2025-01-04', 2, archive)
    assert snapshot == date(2025, 1, 3)
    assert members['Asset'].tolist() == ['C', 'A']
    assert members['Change'].tolist() == ['Added', 'Unchanged']
    assert removed == ['B']


def test_before_archive_does_not_use_future_membership(archive):
    members, snapshot, removed = read_membership('2020-01-01', 2, archive)
    assert members.empty and snapshot is None and not removed
    assert archive_bounds(archive) == (date(2025, 1, 2), date(2025, 1, 3))
