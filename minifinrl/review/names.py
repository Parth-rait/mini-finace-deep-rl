"""Names people type for companies and funds, and assets the review declines."""

from minifinrl.market.universe import CROSS_ASSET_ETFS, SMALL_TICKER

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

# every symbol the name lists know, for suggestions and for reading bare capitals in text
KNOWN_SYMBOLS = sorted(set(SMALL_TICKER) | set(CROSS_ASSET_ETFS) | set(NAME_ALIASES.values()))
