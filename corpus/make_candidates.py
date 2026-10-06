"""Pick candidate texts for labelling from the StockTwits posts dataset.

    python corpus/make_candidates.py              # writes corpus/candidates.jsonl (150 texts)
    python corpus/make_candidates.py --n 300 --seed 1

Source: ElKulako/stocktwits-emoji on Hugging Face (MIT licence; StockTwits
posts about BTC, ETH and SHIB, Nov 2021 to Jun 2022; cite IEEE document
10223689). The validation and test files are downloaded once into
data/raw/stocktwits/ (gitignored).

Kept: posts that read like someone reasoning about a trade (buying, selling,
holding, a loss, the crowd, a past price), 40 to 300 characters, no links.
@-mentions are removed. Duplicates are dropped. The sample is spread evenly
over the dataset's bullish, bearish and neutral files.

No labels are filled in, on purpose: pre-filled guesses steer the person
labelling. Copy the texts you choose into corpus/bias_labels.jsonl and add
"labels" as described in corpus/LABELLING.md. Both files stay out of git.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "stocktwits"
BASE = "https://huggingface.co/datasets/ElKulako/stocktwits-emoji/resolve/main/"
FILES = {"bull": ["val_bull.txt", "test-emoji-bull.txt"], "bear": ["val_bear.txt", "test_emoji_bear.txt"],
         "neutral": ["val_net.txt", "test-emoji-net.txt"]}
SOURCE = "ElKulako/stocktwits-emoji (MIT), StockTwits"

# phrases that suggest a trade decision or the reasoning behind one
CUES = re.compile(
    r"\b(buy|buying|bought|sell|selling|sold|hold|holding|hodl\w*|dip|entry|exit|position|loss|losses|profit|"
    r"stop.?loss|all in|average down|averaging|dca|short|long|calls|puts|break ?even|bag ?hold\w*|"
    r"everyone|everybody|whole (sub|group|chat)|fomo|miss(ing)? out|win it back|make it back|can'?t lose|"
    r"guaranteed|was at|used to be|back to)\b",
    re.IGNORECASE,
)
URL = re.compile(r"https?://|www\.", re.IGNORECASE)
MENTION = re.compile(r"@\w+")


def download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "mini-finrl/0.2"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", MENTION.sub("", text)).strip()


def keep(text: str) -> bool:
    return 40 <= len(text) <= 300 and len(text.split()) >= 6 and not URL.search(text) and bool(CUES.search(text))


def candidates(lines_by_kind: dict[str, list[tuple[str, int, str]]], n: int, seed: int) -> list[dict]:
    """lines_by_kind: kind -> [(file, line number, raw text)]. Even split over kinds."""
    rng = random.Random(seed)
    seen, pools = set(), {}
    for kind, lines in lines_by_kind.items():
        pool = []
        for file, lineno, raw in lines:
            text = clean(raw)
            norm = re.sub(r"\W+", " ", text.lower()).strip()
            if keep(text) and norm not in seen:
                seen.add(norm)
                pool.append((file, lineno, text))
        rng.shuffle(pool)
        pools[kind] = pool
    out, per_kind = [], n // len(pools)
    for kind, pool in pools.items():
        for file, lineno, text in pool[:per_kind]:
            out.append({"id": f"st-{len(out) + 1:04d}", "text": text, "source": f"{SOURCE}, {file} line {lineno}"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "corpus" / "candidates.jsonl")
    args = ap.parse_args()
    lines = {kind: [(f, i, line) for f in files for i, line in enumerate(download(f).read_text().splitlines(), 1)]
             for kind, files in FILES.items()}
    rows = candidates(lines, args.n, args.seed)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} candidates to {args.out} (from {sum(len(v) for v in lines.values())} posts)")


if __name__ == "__main__":
    main()
