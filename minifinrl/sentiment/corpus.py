"""
StockTwits posts with the author's own tag (bullish, bearish, or none), for
measuring how well a reader gets the mood.

Source: ElKulako/stocktwits-emoji on Hugging Face (MIT licence; StockTwits
posts about BTC, ETH and SHIB, Nov 2021 to Jun 2022; cite IEEE document
10223689). Files are downloaded once into data/raw/stocktwits/ (gitignored).
The texts stay local; only scores are recorded.

The tag is the author's own Bullish/Bearish button, so it is a label of the
author's stance, not of the text alone, and "neutral" means "no tag", which
includes plenty of opinionated posts. Scores against it are a floor-level
check of mood reading, not a ceiling.
"""

from __future__ import annotations

import random
import re
import urllib.request
from pathlib import Path

from minifinrl.platform.settings import DATA_RAW, HTTP_TIMEOUT_S

CACHE = Path(DATA_RAW) / "stocktwits"
BASE = "https://huggingface.co/datasets/ElKulako/stocktwits-emoji/resolve/main/"
# held-out test files; the validation files were used to pick labelling candidates (corpus/make_candidates.py)
TEST_FILES = {"bullish": "test-emoji-bull.txt", "bearish": "test_emoji_bear.txt", "neutral": "test-emoji-net.txt"}
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")


def _file(name: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "mini-finrl/0.2"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S * 3) as r:
            path.write_bytes(r.read())
    return path


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", _MENTION.sub("", text)).strip()


def mood_sample(n_per_class: int = 100, seed: int = 0, cache: Path = CACHE) -> list[dict]:
    """[{id, text, mood}], the same number of posts per mood, chosen at random
    (seeded) from posts of 20 to 300 characters without links, duplicates removed."""
    rng = random.Random(seed)
    rows = []
    for mood, name in TEST_FILES.items():
        seen, pool = set(), []
        for i, line in enumerate(_file(name, cache).read_text(encoding="utf-8", errors="replace").splitlines()):
            t = clean(line)
            if 20 <= len(t) <= 300 and not _URL.search(t) and t.lower() not in seen:
                seen.add(t.lower())
                pool.append({"id": f"{name}:{i + 1}", "text": t, "mood": mood})
        if len(pool) < n_per_class:
            raise ValueError(f"only {len(pool)} usable {mood} posts, asked for {n_per_class}")
        rows += rng.sample(pool, n_per_class)
    rng.shuffle(rows)
    return rows
