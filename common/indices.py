"""Read historical market-cap membership without treating market caps as returns."""

from pathlib import Path

import duckdb
import pandas as pd

INDEX_DB = str(Path(__file__).resolve().parent.parent / "data" / "market_cap_data.db")


def archive_bounds(db_path=INDEX_DB):
    with duckdb.connect(db_path, read_only=True) as con:
        return con.execute("SELECT min(date), max(date) FROM market_cap_data").fetchone()


def read_membership(as_of, size=100, db_path=INDEX_DB):
    if not 1 <= size <= 100:
        raise ValueError("Index size must be between 1 and 100.")
    with duckdb.connect(db_path, read_only=True) as con:
        snapshot = con.execute("SELECT max(date) FROM market_cap_data WHERE date <= ?", [str(as_of)]).fetchone()[0]
        if snapshot is None:
            return pd.DataFrame(columns=["Asset", "Source market cap"]), None, []
        prior = con.execute("SELECT max(date) FROM market_cap_data WHERE date < ?", [snapshot]).fetchone()[0]

        def constituents(day):
            if day is None:
                return pd.DataFrame(columns=["Asset", "Source market cap"])
            return con.execute(
                "SELECT symbol AS Asset, market_cap AS \"Source market cap\" FROM market_cap_data "
                "WHERE date = ? AND market_cap > 0 ORDER BY market_cap DESC, symbol LIMIT ?",
                [day, size],
            ).df()

        current = constituents(snapshot)
        previous = constituents(prior)
    current["Change"] = ["Added" if s not in set(previous["Asset"]) else "Unchanged" for s in current["Asset"]]
    removed = sorted(set(previous["Asset"]) - set(current["Asset"]))
    return current, snapshot, removed
