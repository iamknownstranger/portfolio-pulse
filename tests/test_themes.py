import re

from common.themes import (
    MIN_CONSTITUENTS,
    THEMES,
    all_tickers,
    asset_classes,
    theme_label,
    themes_by_asset_class,
)

REQUIRED_KEYS = {"label", "icon", "asset_class", "thesis", "tickers"}
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^]*$")


class TestThemeDefinitions:
    def test_every_theme_has_the_required_fields(self):
        for slug, theme in THEMES.items():
            assert REQUIRED_KEYS <= set(theme), f"{slug} is missing fields"

    def test_every_theme_has_enough_constituents_to_score(self):
        # Breadth and cross-sectional money flow are meaningless on a basket
        # too small to disagree with itself.
        for slug, theme in THEMES.items():
            assert len(theme["tickers"]) >= MIN_CONSTITUENTS, slug

    def test_tickers_are_yahoo_shaped(self):
        for slug, theme in THEMES.items():
            for ticker in theme["tickers"]:
                assert TICKER_PATTERN.match(ticker), f"{slug}: {ticker}"

    def test_no_duplicate_tickers_within_a_theme(self):
        for slug, theme in THEMES.items():
            assert len(theme["tickers"]) == len(set(theme["tickers"])), slug

    def test_labels_and_theses_are_populated(self):
        for slug, theme in THEMES.items():
            assert theme["label"].strip(), slug
            assert theme["thesis"].strip(), slug

    def test_slugs_are_lowercase_identifiers(self):
        for slug in THEMES:
            assert re.match(r"^[a-z0-9_]+$", slug), slug


class TestAllTickers:
    def test_deduplicates_across_themes(self):
        tickers = all_tickers()
        assert len(tickers) == len(set(tickers))

    def test_covers_every_theme_constituent(self):
        tickers = set(all_tickers())
        for theme in THEMES.values():
            assert set(theme["tickers"]) <= tickers

    def test_universe_stays_small_enough_for_one_batched_download(self):
        # The page depends on fetching the whole universe in a single call.
        assert len(all_tickers()) <= 150


class TestAssetClassFilters:
    def test_asset_classes_are_sorted_and_unique(self):
        classes = asset_classes()
        assert classes == sorted(set(classes))

    def test_filtering_returns_only_that_class(self):
        for asset_class in asset_classes():
            subset = themes_by_asset_class(asset_class)
            assert subset
            assert all(t["asset_class"] == asset_class for t in subset.values())

    def test_none_and_all_return_everything(self):
        assert len(themes_by_asset_class(None)) == len(THEMES)
        assert len(themes_by_asset_class("All")) == len(THEMES)

    def test_unknown_class_returns_nothing(self):
        assert themes_by_asset_class("Real Estate") == {}

    def test_filtering_does_not_mutate_the_registry(self):
        themes_by_asset_class(None).clear()
        assert THEMES


class TestThemeLabel:
    def test_includes_icon_and_label(self):
        for slug, theme in THEMES.items():
            label = theme_label(slug)
            assert theme["icon"] in label
            assert theme["label"] in label

    def test_unknown_slug_falls_back_to_the_slug(self):
        assert theme_label("does_not_exist") == "does_not_exist"
