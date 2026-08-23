"""Tests for the EDGAR parsing layer. Nothing here touches the network."""

import pandas as pd
import pytest

from common.sec_edgar import (
    CURATED_MANAGERS,
    DEFAULT_CONTACT_EMAIL,
    build_issuer_ticker_map,
    diff_positions,
    map_positions_to_tickers,
    normalize_issuer_name,
    normalize_position_values,
    parse_form4_xml,
    parse_infotable_xml,
    summarize_insider_buys,
    _user_agent,
)

# Shape mirrors a real ownership form: transaction fields are wrapped in a
# <value> child, which is what makes the parsing non-obvious.
FORM4_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>MP</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Rosenthal Michael Stuart</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Operating Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-09</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>54.30</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-10</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1262</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-11</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>60.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

# Real 13F info tables are namespaced; the parser must be namespace-agnostic.
INFOTABLE_XML = b"""<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>NVIDIA CORPORATION</nameOfIssuer>
    <cusip>67066G104</cusip>
    <value>773586958</value>
    <shrsOrPrnAmt><sshPrnamt>4000000</sshPrnamt></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>BROADCOM INC</nameOfIssuer>
    <cusip>11135F101</cusip>
    <value>497845413</value>
    <shrsOrPrnAmt><sshPrnamt>1500000</sshPrnamt></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>NVIDIA CORPORATION</nameOfIssuer>
    <cusip>67066G104</cusip>
    <value>1000000</value>
    <shrsOrPrnAmt><sshPrnamt>5000</sshPrnamt></shrsOrPrnAmt>
  </infoTable>
</informationTable>
"""


class TestUserAgent:
    def test_declares_a_contact_address(self):
        # SEC rejects a User-Agent that carries a URL instead of an email.
        assert "@" in _user_agent()
        assert "http" not in _user_agent()

    def test_deployer_can_override_the_contact(self, monkeypatch):
        monkeypatch.setenv("SEC_CONTACT_EMAIL", "ops@fund.test")
        assert "ops@fund.test" in _user_agent()

    def test_falls_back_to_the_placeholder_when_unset(self, monkeypatch):
        monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
        assert DEFAULT_CONTACT_EMAIL in _user_agent()

    def test_blank_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("SEC_CONTACT_EMAIL", "   ")
        assert DEFAULT_CONTACT_EMAIL in _user_agent()


class TestParseForm4:
    def test_extracts_every_non_derivative_transaction(self):
        assert len(parse_form4_xml(FORM4_XML)) == 3

    def test_unwraps_nested_value_elements(self):
        # The wrapper elements carry no text of their own; reading them
        # directly yields blanks and silently drops every transaction.
        buy = parse_form4_xml(FORM4_XML)[0]
        assert buy["shares"] == 10000.0
        assert buy["price"] == pytest.approx(54.30)
        assert buy["acquired"] == "A"
        assert buy["transaction_date"] == "2026-06-09"

    def test_computes_transaction_value(self):
        assert parse_form4_xml(FORM4_XML)[0]["value"] == pytest.approx(543000.0)

    def test_carries_issuer_and_owner_identity(self):
        buy = parse_form4_xml(FORM4_XML)[0]
        assert buy["ticker"] == "MP"
        assert buy["owner"] == "Rosenthal Michael Stuart"
        assert buy["role"] == "Chief Operating Officer"

    def test_malformed_xml_returns_empty(self):
        assert parse_form4_xml(b"<not-xml") == []

    def test_empty_payload_returns_empty(self):
        assert parse_form4_xml(None) == []
        assert parse_form4_xml(b"") == []


class TestSummarizeInsiderBuys:
    def test_keeps_only_open_market_purchases(self):
        buys = summarize_insider_buys(parse_form4_xml(FORM4_XML))
        assert len(buys) == 1
        assert buys.iloc[0]["value"] == pytest.approx(543000.0)

    def test_excludes_grants_and_sales(self):
        # Codes A/M/F are compensation mechanics and vastly outnumber real
        # purchases, so including them would drown out the signal.
        buys = summarize_insider_buys(parse_form4_xml(FORM4_XML))
        assert buys["value"].tolist() == [pytest.approx(543000.0)]

    def test_since_filter_drops_older_transactions(self):
        buys = summarize_insider_buys(parse_form4_xml(FORM4_XML), since="2026-07-01")
        assert buys.empty

    def test_no_transactions_returns_empty_frame(self):
        assert summarize_insider_buys([]).empty

    def test_result_is_sorted_newest_first(self):
        doubled = parse_form4_xml(FORM4_XML) + [{
            "ticker": "MP", "owner": "Someone", "role": "Director", "code": "P",
            "acquired": "A", "shares": 1.0, "price": 1.0, "value": 1.0,
            "transaction_date": "2026-08-01",
        }]
        dates = summarize_insider_buys(doubled)["transaction_date"]
        assert dates.is_monotonic_decreasing


class TestParseInfotable:
    def test_parses_namespaced_positions(self):
        positions = parse_infotable_xml(INFOTABLE_XML)
        assert set(positions["cusip"]) == {"67066G104", "11135F101"}

    def test_aggregates_duplicate_issuer_rows(self):
        # Managers report the same issuer once per account; the page wants
        # one line per position.
        positions = parse_infotable_xml(INFOTABLE_XML)
        nvda = positions[positions["cusip"] == "67066G104"].iloc[0]
        assert nvda["value"] == pytest.approx(774586958.0)
        assert nvda["shares"] == pytest.approx(4005000.0)

    def test_malformed_xml_returns_empty_frame(self):
        assert parse_infotable_xml(b"<broken").empty

    def test_empty_payload_returns_empty_frame(self):
        assert parse_infotable_xml(None).empty


class TestNormalizeIssuerName:
    @pytest.mark.parametrize("left,right", [
        ("NVIDIA CORPORATION", "NVIDIA Corp"),
        ("BROADCOM INC", "Broadcom Inc."),
        ("ALPHABET INC CLASS A", "Alphabet Inc."),
        ("ELI LILLY & CO", "ELI LILLY & Co"),
        ("MICRON TECHNOLOGY INC", "Micron Technology, Inc."),
    ])
    def test_spelling_variants_collapse_together(self, left, right):
        assert normalize_issuer_name(left) == normalize_issuer_name(right)

    def test_distinct_companies_stay_distinct(self):
        assert normalize_issuer_name("NVIDIA CORP") != normalize_issuer_name("BROADCOM INC")

    def test_empty_input_returns_empty_string(self):
        assert normalize_issuer_name(None) == ""
        assert normalize_issuer_name("") == ""


class TestPositionMapping:
    @pytest.fixture
    def issuer_map(self):
        return build_issuer_ticker_map({
            "NVDA": "NVIDIA CORP", "AVGO": "Broadcom Inc.", "MU": "Micron Technology, Inc.",
        })

    def test_resolves_known_issuers_to_tickers(self, issuer_map):
        mapped = map_positions_to_tickers(parse_infotable_xml(INFOTABLE_XML), issuer_map)
        assert set(mapped.dropna(subset=["ticker"])["ticker"]) == {"NVDA", "AVGO"}

    def test_unmatched_issuers_keep_a_null_ticker(self):
        mapped = map_positions_to_tickers(
            parse_infotable_xml(INFOTABLE_XML), build_issuer_ticker_map({"MU": "Micron"}),
        )
        # Rows survive with no ticker so unmapped value stays visible.
        assert len(mapped) == 2
        assert mapped["ticker"].isna().all()

    def test_empty_positions_return_a_ticker_column(self):
        empty = pd.DataFrame(columns=["issuer", "cusip", "value", "shares"])
        assert "ticker" in map_positions_to_tickers(empty, {}).columns


class TestNormalizePositionValues:
    def _positions(self):
        return pd.DataFrame({"issuer": ["A"], "cusip": ["X"], "value": [1000.0], "shares": [1.0]})

    def test_pre_2023_values_scale_from_thousands(self):
        import datetime
        scaled = normalize_position_values(self._positions(), datetime.date(2022, 6, 30))
        assert scaled["value"].iloc[0] == pytest.approx(1_000_000.0)

    def test_modern_values_are_left_in_dollars(self):
        import datetime
        scaled = normalize_position_values(self._positions(), datetime.date(2026, 6, 30))
        assert scaled["value"].iloc[0] == pytest.approx(1000.0)

    def test_missing_report_date_leaves_values_alone(self):
        assert normalize_position_values(self._positions(), None)["value"].iloc[0] == 1000.0


class TestDiffPositions:
    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["issuer", "cusip", "ticker", "value"])

    def test_absent_prior_position_is_new(self):
        diff = diff_positions(
            self._frame([["N", "1", "NVDA", 100.0]]), self._frame([]),
        )
        assert diff.iloc[0]["action"] == "New"

    def test_increase_is_added(self):
        diff = diff_positions(
            self._frame([["N", "1", "NVDA", 150.0]]),
            self._frame([["N", "1", "NVDA", 100.0]]),
        )
        assert diff.iloc[0]["action"] == "Added"
        assert diff.iloc[0]["value_change"] == pytest.approx(50.0)

    def test_decrease_is_trimmed(self):
        diff = diff_positions(
            self._frame([["N", "1", "NVDA", 60.0]]),
            self._frame([["N", "1", "NVDA", 100.0]]),
        )
        assert diff.iloc[0]["action"] == "Trimmed"

    def test_unchanged_is_held(self):
        diff = diff_positions(
            self._frame([["N", "1", "NVDA", 100.0]]),
            self._frame([["N", "1", "NVDA", 100.0]]),
        )
        assert diff.iloc[0]["action"] == "Held"

    def test_empty_current_returns_empty(self):
        assert diff_positions(pd.DataFrame(), self._frame([])).empty


class TestCuratedManagers:
    def test_every_entry_has_a_numeric_cik_and_name(self):
        for cik, name in CURATED_MANAGERS.items():
            assert isinstance(cik, int) and 0 < cik < 10**10
            assert isinstance(name, str) and name

    def test_ciks_are_unique(self):
        assert len(CURATED_MANAGERS) == len(set(CURATED_MANAGERS))
