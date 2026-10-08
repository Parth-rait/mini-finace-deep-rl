"""
Market-data providers behind one interface.

PROVENANCE: original to this project. `market/download.py` (adapted from FinRL's
Yahoo downloader) stays as the research path the trained models were built
on; this module is what the incremental store (`market/store.py`) and the
service fetch through, so swapping Yahoo for an official API is one entry
in PRICE_PROVIDERS, not a change to every caller.

Contract every provider keeps:
- the date range is [start, end): end is exclusive, same as yfinance
- prices come back UNADJUSTED, with the provider's adjusted close alongside
  in `adj_close`. Adjusting is done once, at read time, in market/store.py.
  Storing already-adjusted prices would make incremental appends wrong:
  back-adjusted history is rescaled every time a dividend is paid.
- a network/provider failure raises ProviderError (the service turns it
  into 503 + Retry-After); "no data for this range" is an empty frame.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Protocol

import pandas as pd

from minifinrl.platform.log import get_logger
from minifinrl.market.settings import FRED_VIX_SERIES, VIX_TICKER
from minifinrl.platform.settings import HTTP_RETRIES, HTTP_TIMEOUT_S

log = get_logger(__name__)

PROVIDER_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


class ProviderError(RuntimeError):
    """The provider could not be reached or refused the request. Transient
    from the caller's point of view: retry later, don't train on it."""

    error_kind = "unavailable"


class ProviderNotConfigured(ProviderError):
    """No API key, or the key was rejected. Not retried: retrying won't help."""


class PriceProvider(Protocol):
    name: str

    def fetch(self, tic: str, start: str, end: str) -> pd.DataFrame:
        """Daily bars for `tic` in [start, end), columns PROVIDER_COLUMNS,
        date as 'YYYY-MM-DD' strings, sorted ascending."""
        ...


class SeriesProvider(Protocol):
    name: str

    def fetch(self, start: str, end: str) -> pd.Series:
        """One daily series in [start, end), indexed by 'YYYY-MM-DD'."""
        ...


def with_retries(
    fn: Callable[[], object],
    *,
    what: str,
    retries: int = HTTP_RETRIES,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
):
    """Call `fn`, retrying ProviderError with exponential backoff (1s, 2s,
    4s, ...). Anything else propagates immediately: a bug is not transient."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except ProviderNotConfigured:
            raise
        except ProviderError as exc:
            if attempt == retries:
                log.error("%s: giving up after %d attempts: %s", what, attempt, exc)
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            log.warning("%s: attempt %d/%d failed (%s), retrying in %.0fs", what, attempt, retries, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=PROVIDER_COLUMNS)


# ---- Yahoo (research default; scraped, personal-use terms) ------------------


class YahooProvider:
    name = "yahoo"

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        from minifinrl.market.download import _flatten_columns

        try:
            raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        except Exception as exc:  # yfinance raises its own rate-limit/network types
            raise ProviderError(f"yahoo {symbol}: {type(exc).__name__}: {exc}") from exc
        if raw is None or raw.empty:
            return pd.DataFrame()
        return _flatten_columns(raw).reset_index()

    def fetch(self, tic: str, start: str, end: str) -> pd.DataFrame:
        raw = with_retries(lambda: self._download(tic, start, end), what=f"yahoo {tic}")
        if raw.empty:
            return _empty()
        raw = raw.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        if "adj_close" not in raw.columns:
            raw["adj_close"] = raw["close"]
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        return raw[PROVIDER_COLUMNS].sort_values("date").reset_index(drop=True)


class YahooVixProvider:
    name = "yahoo"

    def fetch(self, start: str, end: str) -> pd.Series:
        bars = YahooProvider().fetch(VIX_TICKER, start, end)
        return bars.set_index("date")["close"].rename("vix").astype(float)


# ---- FRED (official, free; VIX = CBOE close, series VIXCLS) ------------------


def _http_get(url: str, *, timeout: float = HTTP_TIMEOUT_S, headers: dict[str, str] | None = None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "mini-finrl/0.1", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # 4xx other than 429 is our fault (bad series id, bad key): not
        # worth retrying, and not a provider outage either.
        if exc.code in (401, 403):
            raise ProviderNotConfigured(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}: API key missing or rejected") from exc
        if exc.code == 429 or exc.code >= 500:
            raise ProviderError(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}") from exc
        raise ValueError(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}: check the request") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"{type(exc).__name__}: {exc}") from exc


class FredSeriesProvider:
    """With FRED_API_KEY in the environment, uses the documented API
    (api.stlouisfed.org). Without it, the public fredgraph CSV download of
    the same series. Missing observations (FRED writes '.') are dropped, so
    market holidays simply have no row, same as Yahoo."""

    name = "fred"

    def __init__(self, series_id: str = FRED_VIX_SERIES, *, name: str = "vix",
                 http_get: Callable[[str], str] = _http_get):
        self.series_id = series_id
        self.series_name = name
        self._get = http_get

    def _url(self, start: str, end_inclusive: str) -> tuple[str, str]:
        key = os.environ.get("FRED_API_KEY")
        if key:
            q = urllib.parse.urlencode(
                {
                    "series_id": self.series_id,
                    "api_key": key,
                    "file_type": "json",
                    "observation_start": start,
                    "observation_end": end_inclusive,
                }
            )
            return f"https://api.stlouisfed.org/fred/series/observations?{q}", "json"
        q = urllib.parse.urlencode({"id": self.series_id, "cosd": start, "coed": end_inclusive})
        return f"https://fred.stlouisfed.org/graph/fredgraph.csv?{q}", "csv"

    @staticmethod
    def _parse_csv(text: str) -> list[tuple[str, str]]:
        rows = list(csv.reader(io.StringIO(text)))
        if not rows or len(rows[0]) < 2:
            raise ProviderError("fredgraph returned no CSV header")
        return [(r[0], r[1]) for r in rows[1:] if len(r) >= 2]

    @staticmethod
    def _parse_json(text: str) -> list[tuple[str, str]]:
        payload = json.loads(text)
        if "observations" not in payload:
            raise ProviderError(f"FRED API error: {payload.get('error_message', payload)}")
        return [(o["date"], o["value"]) for o in payload["observations"]]

    def fetch(self, start: str, end: str) -> pd.Series:
        # FRED ranges are inclusive; ours are [start, end).
        end_inclusive = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        url, fmt = self._url(start, end_inclusive)
        text = with_retries(lambda: self._get(url), what=f"fred {self.series_id}")
        pairs = self._parse_json(text) if fmt == "json" else self._parse_csv(text)

        values = pd.to_numeric(pd.Series({d: v for d, v in pairs}, dtype=object), errors="coerce")
        values = values.dropna().astype(float)
        values.index = pd.to_datetime(values.index).strftime("%Y-%m-%d")
        values = values[(values.index >= start) & (values.index < end)].sort_index()
        values.index.name = "date"
        log.info("fred %s: %d observations %s..%s", self.series_id, len(values), start, end_inclusive)
        return values.rename(self.series_name)


# ---- Tiingo (official REST, free key; raw and adjusted OHLC in one call) -------------


def _require_env(*names: str) -> list[str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise ProviderNotConfigured(f"set {', '.join(missing)} in the environment (never in the repo)")
    return [os.environ[n] for n in names]


def _end_inclusive(end: str) -> str:
    return (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


class TiingoProvider:
    """GET api.tiingo.com/tiingo/daily/{ticker}/prices. Returns unadjusted
    OHLCV and `adjClose` per day, which is exactly the store's contract
    (unadjusted + adj_close). Key: TIINGO_API_KEY."""

    name = "tiingo"
    URL = "https://api.tiingo.com/tiingo/daily/{tic}/prices"

    def __init__(self, *, http_get: Callable[..., str] = _http_get):
        self._get = http_get

    def fetch(self, tic: str, start: str, end: str) -> pd.DataFrame:
        (key,) = _require_env("TIINGO_API_KEY")
        q = urllib.parse.urlencode({"startDate": start, "endDate": _end_inclusive(end), "format": "json"})
        url = self.URL.format(tic=tic.lower()) + "?" + q
        text = with_retries(lambda: self._get(url, headers={"Authorization": f"Token {key}"}), what=f"tiingo {tic}")
        rows = json.loads(text)
        if isinstance(rows, dict):  # Tiingo reports errors as {"detail": ...}
            raise ProviderError(f"tiingo {tic}: {rows.get('detail', rows)}")
        if not rows:
            return _empty()
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "date": pd.to_datetime(df["date"], utc=True).dt.strftime("%Y-%m-%d"),
            "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
            "adj_close": df["adjClose"], "volume": df["volume"],
        })
        return out[(out["date"] >= start) & (out["date"] < end)].sort_values("date").reset_index(drop=True)


# ---- Alpaca market data (official REST, free key, IEX feed) ---------------------------


class AlpacaProvider:
    """GET data.alpaca.markets/v2/stocks/bars, daily bars, paginated.
    Two requests per range: adjustment=raw (the OHLC we store) and
    adjustment=all (split+dividend adjusted; its close becomes adj_close).
    Bar timestamps are midnight New York time, so the date is taken in
    America/New_York, not UTC. Keys: ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY.

    The free feed is IEX: closes are IEX's last trade, not the consolidated
    official close, and volume is IEX-only. Use crosscheck_prices to measure
    the gap against another source before trusting it for research.
    """

    name = "alpaca"
    URL = "https://data.alpaca.markets/v2/stocks/bars"

    def __init__(self, *, feed: str = "iex", http_get: Callable[..., str] = _http_get):
        self.feed = feed
        self._get = http_get

    def _bars(self, tic: str, start: str, end: str, adjustment: str, headers: dict) -> list[dict]:
        bars, token = [], None
        while True:
            params = {"symbols": tic, "timeframe": "1Day", "start": start, "end": _end_inclusive(end),
                      "adjustment": adjustment, "feed": self.feed, "limit": 10000}
            if token:
                params["page_token"] = token
            url = self.URL + "?" + urllib.parse.urlencode(params)
            payload = json.loads(with_retries(lambda: self._get(url, headers=headers), what=f"alpaca {tic} {adjustment}"))
            bars += (payload.get("bars") or {}).get(tic, [])
            token = payload.get("next_page_token")
            if not token:
                return bars

    def fetch(self, tic: str, start: str, end: str) -> pd.DataFrame:
        key_id, secret = _require_env("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")
        headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        raw, adj = self._bars(tic, start, end, "raw", headers), self._bars(tic, start, end, "all", headers)
        if not raw:
            return _empty()

        def frame(bars):
            d = pd.DataFrame(bars)
            d["date"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            return d.set_index("date")

        r, a = frame(raw), frame(adj)
        if not r.index.equals(a.index):
            raise ProviderError(f"alpaca {tic}: raw and adjusted bars cover different dates")
        out = pd.DataFrame({"date": r.index, "open": r["o"].values, "high": r["h"].values, "low": r["l"].values,
                            "close": r["c"].values, "adj_close": a["c"].values, "volume": r["v"].values})
        return out[(out["date"] >= start) & (out["date"] < end)].sort_values("date").reset_index(drop=True)


PRICE_PROVIDERS: dict[str, Callable[[], PriceProvider]] = {
    "yahoo": YahooProvider,
    "tiingo": TiingoProvider,
    "alpaca": AlpacaProvider,
}
VIX_PROVIDERS: dict[str, Callable[[], SeriesProvider]] = {"fred": FredSeriesProvider, "yahoo": YahooVixProvider}


def get_price_provider(name: str) -> PriceProvider:
    if name not in PRICE_PROVIDERS:
        raise ValueError(f"unknown price provider '{name}', expected one of {list(PRICE_PROVIDERS)}")
    return PRICE_PROVIDERS[name]()


# 3-month Treasury bill, secondary market, % per year (E03: the risk-free rate in Sharpe/Sortino)
RATE_PROVIDERS: dict[str, Callable[[], SeriesProvider]] = {
    "fred": lambda: FredSeriesProvider("DTB3", name="rate"),
}


def get_rate_provider(name: str) -> SeriesProvider:
    if name not in RATE_PROVIDERS:
        raise ValueError(f"unknown rate provider '{name}', expected one of {list(RATE_PROVIDERS)}")
    return RATE_PROVIDERS[name]()


def get_vix_provider(name: str) -> SeriesProvider:
    if name not in VIX_PROVIDERS:
        raise ValueError(f"unknown VIX provider '{name}', expected one of {list(VIX_PROVIDERS)}")
    return VIX_PROVIDERS[name]()
