# Labelling trading texts for bias signals

The bias classifier is scored against texts labelled by hand. This file says
what to collect, how to label it, and how to run the evaluation.

## What to collect

About 50 short texts in which someone explains a trade or an intention to
trade: forum posts, journal entries, comments. Use only sources whose terms
allow reuse (a published dataset with a licence, or text you wrote yourself),
and remove usernames and anything that identifies a person. Note the source
of each text in the `source` field.

You label the text, not the person who wrote it.

## The six labels

A text can have several labels, or none. Use the empty list `[]` when no bias
is signalled; those texts matter as much as the others.

| label | the text signals... | example phrasing |
|---|---|---|
| `fomo` | urgency to enter because others are already gaining | "everyone is up 40% on this, I have to get in" |
| `revenge_trading` | trading to win back a recent loss | "lost big yesterday, doubling down to make it back" |
| `overconfidence` | certainty about the outcome, risk ignored | "can't lose", "all in" |
| `loss_aversion` | refusing to sell at a loss, waiting to break even | "I'll hold until it's back to what I paid" |
| `herding` | doing something because the crowd is doing it | "the whole sub is buying" |
| `anchoring` | judging the price against an arbitrary past reference | "it was $300 last year, so $200 is cheap" |

FOMO and herding often appear together. Label `fomo` when the reason given is
fear of missing gains, `herding` when the reason given is what other people
are doing, and both when the text gives both reasons.

Four rules, settled while labelling the first 50 texts:

- Only the author's own reasoning counts. A bias the author describes in, or
  attributes to, other people ("most people here need a 1000% pump to get back
  to even") is not labelled.
- Sarcasm or mockery of an attitude is not that attitude ("buy the dip bro,
  trust me bro" is `[]`).
- Going against the crowd is not herding ("too many bears here, perfect time
  to buy").
- A future price target is not anchoring. Anchoring needs a past reference:
  a past price, the purchase price, a previous high.

The LLM classifier is given these same definitions and rules, and none of the
labelled texts.

References: Kahneman and Tversky (1979) for loss aversion; Tversky and
Kahneman (1974) for anchoring; Shefrin and Statman (1985) for the disposition
effect; Bikhchandani, Hirshleifer and Welch (1992) for herding; Barber and
Odean (2001) for overconfidence in trading.

## File format

`corpus/bias_labels.jsonl`, one JSON object per line:

```json
{"id": "t001", "text": "...", "labels": ["fomo"], "labels_b": ["fomo", "herding"], "source": "..."}
```

`labels` is the reference. `labels_b` is optional: a second person's labels
for the same text, made without seeing the first person's. If at least some
texts have `labels_b`, the evaluation also reports how much the two people
agree, which tells you how hard the task is before you judge a model on it.
Lines starting with `//` are ignored. `bias_labels.template.jsonl` shows the
format with three made-up examples; copy it and replace them.

## Running the evaluation

```bash
python -m minifinrl evaluate-biases                       # default path above
python -m minifinrl evaluate-biases --labels-path other.jsonl
```

It reports precision, recall, F1 and Cohen's kappa for each label, the macro
averages, and the exact-match rate. Kappa corrects agreement for chance: 0 is
chance level, 1 is perfect, and values above about 0.6 are usually read as
substantial. A label that nobody used gets no kappa.
