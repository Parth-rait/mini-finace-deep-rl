"""
What to say before a trade, as plain rules over the state reading, the bias
signals and the outcome range. Every nudge is about how the decision is being
made, never about whether the stock will go up: the model has no edge for
that, and saying so would be advice.
"""

from __future__ import annotations

from minifinrl.plan.schemas import Nudge
from minifinrl.regime.outlook import Outlook
from minifinrl.review.schemas import BiasSpan
from minifinrl.sentiment.ports import StateReading


def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%" if abs(x) >= 0.005 else f"{x * 100:+.1f}%"


# one line per signal or bias: the habit it points to, said kindly
BY_SIGNAL = {
    "urgency": "Nothing about this trade has to happen today. If the reason holds, it will still hold tomorrow.",
    "herd": "Other people buying is not a reason on its own. What do you know that the price doesn't already show?",
    "certainty": "No trade is certain. Picture the low end of the range below happening, and decide now what you'd do.",
    "greed": "Excitement about big gains is when people size up too much. Decide the amount you could lose without it hurting.",
    "fear": "Fear makes for rushed entries and exits. Write your exit rule while you're calm, then follow it.",
    "regret": "Missing an earlier move isn't a reason to chase this one. Would you make this trade if you'd never seen the last run?",
    "frustration": "Trading while annoyed at the market tends to raise risk. A short break before deciding usually costs nothing.",
}
BY_BIAS = {
    "fomo": "BY_SIGNAL:urgency",
    "herding": "BY_SIGNAL:herd",
    "overconfidence": "BY_SIGNAL:certainty",
    "revenge_trading": "Trying to win back a loss is a common way to turn one loss into two. Treat this trade on its own merits.",
    "loss_aversion": "Decide your exit at a loss before you enter, so the decision isn't made under pressure later.",
    "anchoring": "A past price is not a target the stock owes you. What is the case for the price from here?",
}


def plan_nudges(risk, outlook: Outlook | None, has_stop: bool) -> list[Nudge]:
    """About the plan's numbers: no stop, a stop inside ordinary noise, more risk than reward, a big bite of the account."""
    out: list[Nudge] = []
    if not has_stop:
        out.append(Nudge(level="caution", text="There's no stop yet. Deciding in advance where you're wrong is the part of a plan "
                                               "people skip most, and the one that limits a bad trade."))
        return out
    if outlook is not None and outlook.prob_stop is not None and outlook.prob_stop >= 0.5:
        out.append(Nudge(level="caution", text=(
            f"Your stop was touched in {outlook.prob_stop:.0%} of the normal-luck paths within {outlook.horizon_days} days. "
            "A stop inside ordinary day-to-day movement tends to get hit by noise rather than by the idea being wrong.")))
    if risk is not None and risk.reward_risk is not None and risk.reward_risk < 1:
        out.append(Nudge(level="caution", text=(
            f"You'd risk more than you aim to make ({risk.reward_risk:.1f} : 1). That only works if you're right well over "
            "half the time.")))
    if risk is not None and risk.account_risk_pct is not None and risk.account_risk_pct > 0.02:
        out.append(Nudge(level="caution", text=(
            f"If the stop is hit you lose {risk.account_risk_pct:.1%} of your account on one idea. A common practice is to keep "
            "that to 1 or 2%, so a run of losses can't do lasting damage.")))
    return out


def build(state: StateReading | None, biases: list[BiasSpan], outlook: Outlook | None, ticker: str,
          has_reasoning: bool, extra: list[Nudge] | None = None) -> list[Nudge]:
    out: list[Nudge] = []
    if state is not None:
        kinds = [s.kind.value for s in state.signals]
        if state.level == "hot":
            out.append(Nudge(level="pause", text=(
                "Your words show strong emotion or pressure (" + ", ".join(kinds) + "). Trades made in this state are the "
                "ones people most often regret. Consider waiting a day, then reading your reason again.")))
        elif state.level == "warm":
            out.append(Nudge(level="caution", text=(
                "There is some pressure in your words (" + ", ".join(kinds) + "). Before you trade, write down what would "
                "make you sell.")))
        seen = set()
        for k in kinds:
            if k in BY_SIGNAL and k not in seen:
                out.append(Nudge(level="caution", text=BY_SIGNAL[k]))
                seen.add(k)
        for b in biases:
            text = BY_BIAS.get(b.label)
            if text and text.startswith("BY_SIGNAL:"):
                k = text.split(":", 1)[1]
                if k in seen:
                    continue
                text, seen = BY_SIGNAL[k], seen | {k}
            if text and text not in {n.text for n in out}:
                out.append(Nudge(level="caution", text=text))
        if state.level == "calm" and not biases:
            out.append(Nudge(level="info", text="Your reason reads calmly, with no signs of pressure or the common biases."))
    elif not has_reasoning:
        out.append(Nudge(level="info", text="Add a sentence on why you want to make this trade: the check reads your words for "
                                            "emotion and pressure, which is where most avoidable mistakes start."))
    out += extra or []
    if outlook is not None:
        out.append(Nudge(level="info", text=(
            f"In today's {outlook.regime} market, a {outlook.horizon_days}-day hold of {ticker} normally ends between "
            f"{_pct(outlook.p05)} and {_pct(outlook.p95)}; {outlook.prob_loss:.0%} of the simulated paths lost money. "
            f"Would you be fine at {_pct(outlook.p05)}?")))
    return out
