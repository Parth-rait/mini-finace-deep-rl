"""Ticker universes. Kept separate from settings.py so adding a bigger
universe later doesn't require touching the tunables file."""

SMALL_TICKER = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "JNJ",
    "XOM",
]

# E06: cross-asset universe. All 12 trade since before 2014-01-02 (checked
# 6 Oct 2026), so a 2014 start has no gaps and, being ETFs, no survivorship
# bias. XLC (2018) and XLRE (2015) are excluded for that reason.
CROSS_ASSET_ETFS = [
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",  # 9 original SPDR sectors
    "TLT", "IEF",  # long and intermediate US Treasuries
    "GLD",  # gold
]

UNIVERSES = {"mega8": SMALL_TICKER, "cross_asset12": CROSS_ASSET_ETFS}

# ---- trade review: names people write, and assets the review can't model ----------------
# Company and fund names as people type them -> exchange symbol. Used to read
# free-text trades and to suggest matches; any other valid US symbol also
# works if Yahoo has its history.
NAME_ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META", "jpmorgan": "JPM", "jp morgan": "JPM", "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ", "exxon": "XOM", "exxonmobil": "XOM", "tesla": "TSLA", "nvidia": "NVDA",
    "netflix": "NFLX", "amd": "AMD", "intel": "INTC", "coca-cola": "KO", "coca cola": "KO", "coke": "KO",
    "pepsi": "PEP", "walmart": "WMT", "disney": "DIS", "boeing": "BA", "nike": "NKE", "gamestop": "GME",
    "amc": "AMC", "palantir": "PLTR", "coinbase": "COIN", "berkshire": "BRK-B", "visa": "V",
    "mastercard": "MA", "bank of america": "BAC", "goldman": "GS", "goldman sachs": "GS", "pfizer": "PFE",
    "uber": "UBER", "shopify": "SHOP", "spotify": "SPOT", "starbucks": "SBUX", "mcdonalds": "MCD",
    "mcdonald's": "MCD", "costco": "COST", "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE",
    "s&p 500": "SPY", "s&p": "SPY", "sp500": "SPY", "nasdaq": "QQQ", "nasdaq 100": "QQQ", "gold": "GLD",
    "treasuries": "TLT", "bonds": "TLT",
}

# Crypto trades around the clock and isn't driven by the equity-market regime
# the luck test models, so the review declines it rather than answer wrongly.
CRYPTO = {
    "BTC", "ETH", "DOGE", "SHIB", "SOL", "XRP", "ADA", "LTC", "LUNA", "BNB", "AVAX", "DOT", "MATIC", "LEASH",
    "BITCOIN", "ETHEREUM", "DOGECOIN", "SHIBA", "SOLANA", "LITECOIN", "CRYPTO",
}
