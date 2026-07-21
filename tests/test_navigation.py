import pathlib
import py_compile

from common.navigation import DEFAULT_PAGE, PAGE_SECTIONS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _all_entries():
    return [entry for entries in PAGE_SECTIONS.values() for entry in entries]


def test_every_registered_page_exists_and_compiles():
    for path, title, icon, url_path in _all_entries():
        script = REPO_ROOT / path
        assert script.is_file(), f"{title}: missing page script {path}"
        py_compile.compile(str(script), doraise=True)


def test_default_page_is_registered():
    assert DEFAULT_PAGE in {path for path, *_ in _all_entries()}


def test_url_paths_are_unique():
    url_paths = [url_path for *_, url_path in _all_entries()]
    assert len(url_paths) == len(set(url_paths))


def test_asset_class_sections_present():
    # Issue #5: navigation must cover equities (linear and non-linear),
    # fixed income, and commodities.
    sections = " ".join(PAGE_SECTIONS.keys()).lower()
    for asset_class in ["linear", "non-linear", "fixed income", "commodities"]:
        assert asset_class in sections
