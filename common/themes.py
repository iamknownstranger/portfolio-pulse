"""Thematic basket definitions for the Smart Money Radar.

Plain data with no imports, so the universe can be inspected and tested
without pandas or a Streamlit runtime.

Each theme is an equal-weighted basket of liquid, Yahoo-Finance-resolvable
symbols that express one narrative. Baskets are deliberately small (6-9
names): enough for breadth to mean something, few enough that a single
delisting does not silently gut the signal.
"""

# slug -> {label, icon, asset_class, thesis, tickers}
THEMES = {
    "ai_compute": {
        "label": "AI & Accelerated Compute",
        "icon": "🧠",
        "asset_class": "Equity",
        "thesis": "Silicon, networking and thermal plumbing for training and inference clusters.",
        "tickers": ["NVDA", "AMD", "AVGO", "TSM", "MU", "MRVL", "ARM", "ANET", "VRT"],
    },
    "ai_software": {
        "label": "AI Software & Agents",
        "icon": "🤖",
        "asset_class": "Equity",
        "thesis": "Application and platform layer monetizing models rather than selling chips.",
        "tickers": ["MSFT", "PLTR", "NOW", "SNOW", "CRM", "DDOG", "MDB"],
    },
    "semi_equipment": {
        "label": "Semiconductor Equipment",
        "icon": "🔬",
        "asset_class": "Equity",
        "thesis": "Toolmakers that get paid on fab capex regardless of which chip designer wins.",
        "tickers": ["ASML", "AMAT", "LRCX", "KLAC", "TER", "ONTO"],
    },
    "crypto_equities": {
        "label": "Crypto Equities",
        "icon": "⛓️",
        "asset_class": "Equity",
        "thesis": "Listed proxies for digital-asset adoption: exchanges, miners, treasuries.",
        "tickers": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HOOD", "HUT"],
    },
    "crypto_spot": {
        "label": "Digital Assets (Spot)",
        "icon": "🪙",
        "asset_class": "Crypto",
        "thesis": "The underlying tokens themselves — the purest read on crypto risk appetite.",
        "tickers": ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD", "XRP-USD"],
    },
    "nuclear_uranium": {
        "label": "Nuclear & Uranium",
        "icon": "☢️",
        "asset_class": "Equity",
        "thesis": "Fuel cycle and small modular reactors as firm power for datacenter load.",
        "tickers": ["CCJ", "LEU", "SMR", "OKLO", "BWXT", "UEC", "NNE"],
    },
    "power_grid": {
        "label": "Power & Grid Buildout",
        "icon": "⚡",
        "asset_class": "Equity",
        "thesis": "Generation and transmission bottleneck — the physical constraint on AI scaling.",
        "tickers": ["VST", "CEG", "NRG", "GEV", "ETN", "PWR", "POWL"],
    },
    "quantum": {
        "label": "Quantum Computing",
        "icon": "🧬",
        "asset_class": "Equity",
        "thesis": "Pre-revenue optionality; moves violently on research and funding headlines.",
        "tickers": ["IONQ", "RGTI", "QBTS", "QUBT"],
    },
    "space_defense": {
        "label": "Space & Defense Tech",
        "icon": "🚀",
        "asset_class": "Equity",
        "thesis": "Launch, satellite constellations and autonomous defense procurement.",
        "tickers": ["RKLB", "ASTS", "LUNR", "AVAV", "KTOS", "LMT", "RTX"],
    },
    "cybersecurity": {
        "label": "Cybersecurity",
        "icon": "🛡️",
        "asset_class": "Equity",
        "thesis": "Non-discretionary software spend; tends to lead in risk-off software tape.",
        "tickers": ["CRWD", "PANW", "ZS", "S", "FTNT", "OKTA"],
    },
    "robotics": {
        "label": "Robotics & Automation",
        "icon": "🦾",
        "asset_class": "Equity",
        "thesis": "Physical AI: warehouse, surgical and industrial automation.",
        "tickers": ["ISRG", "ROK", "SYM", "TER", "NDSN", "ZBRA"],
    },
    "obesity_biotech": {
        "label": "GLP-1 & Obesity",
        "icon": "💊",
        "asset_class": "Equity",
        "thesis": "Incretin franchise plus the telehealth channel distributing it.",
        "tickers": ["LLY", "NVO", "VKTX", "AMGN", "HIMS", "ZLDPF"],
    },
    "rare_earths": {
        "label": "Rare Earths & Critical Minerals",
        "icon": "⛏️",
        "asset_class": "Equity",
        "thesis": "Supply-chain onshoring of magnet, lithium and copper feedstock.",
        "tickers": ["MP", "ALB", "FCX", "SCCO", "UEC", "TMC"],
    },
    "clean_energy": {
        "label": "Clean Energy & Solar",
        "icon": "🌞",
        "asset_class": "Equity",
        "thesis": "Rate-sensitive renewables; a classic deep-base candidate after long drawdowns.",
        "tickers": ["ENPH", "FSLR", "RUN", "NEE", "SEDG", "ARRY"],
    },
    "evtol": {
        "label": "eVTOL & Advanced Air Mobility",
        "icon": "🛩️",
        "asset_class": "Equity",
        "thesis": "Certification-gated aviation startups; binary newsflow, heavy retail float.",
        "tickers": ["JOBY", "ACHR", "EH", "RCAT"],
    },
    "precious_metals": {
        "label": "Precious Metals & Debasement",
        "icon": "🥇",
        "asset_class": "Commodity",
        "thesis": "Metal plus the miners' operating leverage — the fiat-debasement expression.",
        "tickers": ["GC=F", "SI=F", "NEM", "GOLD", "AEM", "WPM"],
    },
}

# Minimum constituents that must return usable data for a theme to be ranked.
MIN_CONSTITUENTS = 3


def all_tickers():
    """Every unique symbol across all themes, in stable order."""
    seen = {}
    for theme in THEMES.values():
        for ticker in theme["tickers"]:
            seen[ticker] = None
    return list(seen)


def asset_classes():
    """Distinct asset classes present in the universe, sorted."""
    return sorted({theme["asset_class"] for theme in THEMES.values()})


def themes_by_asset_class(asset_class):
    """Themes filtered to one asset class, or all themes when None/'All'."""
    if asset_class in (None, "All"):
        return dict(THEMES)
    return {
        slug: theme
        for slug, theme in THEMES.items()
        if theme["asset_class"] == asset_class
    }


def theme_label(slug):
    """Display label with icon for a theme slug."""
    theme = THEMES.get(slug)
    if theme is None:
        return slug
    return f"{theme['icon']} {theme['label']}"
