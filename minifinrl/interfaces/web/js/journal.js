// Journal: every trade under your ID. Plans, open trades and closed ones, each
// with the state your words were in when you logged it. Plans become trades
// with "I made this trade"; open trades are closed with a sell date.

import { $, $$, el, api, pct, nice, ordinal, today, profile, confirmInline, VERDICT, LABELS, LEVELS } from "./core.js";
import { renderStart } from "./start.js";
import { showReview } from "./review.js";

let filter = "all";

export function showJournal() {
  const pid = profile.get();
  $("#j-gate").hidden = Boolean(pid);
  $("#j-body").hidden = !pid;
  if (!pid) { renderStart($("#j-gate"), () => showJournal()); return; }
  refreshJournal();
}

const num2 = (x) => (x == null ? "" : x.toLocaleString(undefined, { maximumFractionDigits: 2 }));
const money = (x) => x.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });

async function loadInsights(pid) {
  const host = $("#j-insights");
  try {
    const d = await api("/journal-insights", { profile_id: pid });
    const rows = (groups) => groups.filter((g) => g.trades).map((g) =>
      el("tr", {}, el("td", { text: g.label }), el("td", { text: String(g.trades) }),
        el("td", { text: g.avg_return == null ? "-" : pct(g.avg_return) }), el("td", { text: g.win_rate == null ? "-" : `${Math.round(g.win_rate * 100)}%` })));
    const table = (title, groups) => groups.some((g) => g.trades) && el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, ...[title, "Trades", "Average return", "Won"].map((h) => el("th", { text: h })))), el("tbody", {}, ...rows(groups))));
    host.replaceChildren(el("section", { class: "card insights" },
      el("h3", { text: "What your trades say" }),
      ...d.headlines.map((h) => el("p", { class: "headline", text: h })),
      d.needs_attention.length > 0 && el("div", { class: "stack" }, el("p", { class: "eyebrow", text: "Your plan says it's time to act" }),
        el("ul", { class: "attention" }, ...d.needs_attention.map((a) => el("li", {}, el("b", { text: a.ticker }), el("span", { class: "check-badge stop", text: a.label }), el("span", { text: a.message }))))),
      d.closed > 0 && el("div", { class: "grid-2" }, table("By plan", d.by_plan), table("By state when planned", d.by_state))));
    host.hidden = false;
  } catch { host.hidden = true; }
}

export async function refreshJournal() {
  const pid = profile.get(), list = $("#j-list");
  if (!pid || !list) return;
  loadInsights(pid);
  try {
    const d = await api("/list-trades", { profile_id: pid });
    const s = d.summary;
    $("#j-summary").replaceChildren(
      ...[["all", "All", s.count], ["planned", "Plans", s.planned], ["open", "Open", s.open], ["closed", "Closed", s.closed]].map(([key, label, n]) =>
        el("button", { type: "button", class: `filter${filter === key ? " on" : ""}`, "aria-pressed": String(filter === key),
          onclick: () => { filter = key; refreshJournal(); } }, label, el("span", { class: "count", text: String(n) }))));
    const rows = d.trades.filter((t) => filter === "all" || t.status === filter);
    list.replaceChildren(...(rows.length ? rows.map(card)
      : [el("div", { class: "empty-state" }, el("p", { text: d.trades.length ? "Nothing here with this filter." : "No trades yet." }),
          !d.trades.length && el("p", { class: "muted", text: "Check a plan before you trade, or review one you've made. Both can be saved here." }))]));
  } catch (e) {
    if (e.status === 404) { profile.clear(); showJournal(); return; }
    list.replaceChildren(el("p", { class: "muted", text: `Couldn't load your trades: ${e.message}` }));
  }
}

function card(t) {
  const short = t.direction === "short";
  const r = t.result || {};
  const plan = t.plan;
  const pc = r.plan_check;
  const dates = t.status === "planned" ? (t.buy_date ? `Planned for ${nice(t.buy_date)}` : "Planned")
    : `${short ? "Shorted" : "Bought"} ${nice(t.buy_date)} · ${t.sell_date ? `${short ? "covered" : "sold"} ${nice(t.sell_date)}` : "still holding"}`;

  let outcome;
  if (r.status === "ok") {
    const [label, tone] = VERDICT[r.verdict];
    outcome = el("div", { class: "outcome" }, el("b", { class: `ret ${r.your_return >= 0 ? "up" : "down"}`, text: pct(r.your_return) }),
      el("span", { class: `pill ${tone}`, text: label }),
      el("span", { class: "tag", text: `${ordinal(r.percentile)} percentile · ${r.days} days${t.status === "open" ? ` to ${nice(r.exit_date)}` : ""}` }));
  } else if (r.status) {
    outcome = el("p", { class: "muted small", text: r.message || "The review couldn't run." });
  } else if (plan && plan.p05 != null) {
    outcome = el("div", { class: "outcome" }, el("span", { class: "tag", text: `Normal range over ${plan.horizon_days} days:` }),
      el("b", { text: `${pct(plan.p05, 0)} to ${pct(plan.p95, 0)}` }),
      plan.prob_stop != null && el("span", { class: "tag", text: `stop touched in ${Math.round(plan.prob_stop * 100)}% of paths` }));
  } else {
    outcome = el("p", { class: "muted small", text: t.status === "planned" ? "Not checked yet" : "Not reviewed yet" });
  }
  const planLine = t.plan_stop != null && el("p", { class: "plan-line", text:
    `Plan: entry ${num2(t.plan_entry)} · stop ${num2(t.plan_stop)}${t.plan_target ? ` · target ${num2(t.plan_target)}` : ""} · up to ${t.plan_horizon} days`
    + (plan && plan.risk && plan.risk.risk_money != null ? ` · risk ${money(plan.risk.risk_money)}` : "") });
  const checkLine = pc && el("div", { class: "outcome" }, el("span", { class: `check-badge ${pc.verdict}`, text: pc.label }),
    el("span", { class: "small", text: pc.message + (pc.entry_gap != null && Math.abs(pc.entry_gap) >= 0.005 ? ` You entered ${pct(pc.entry_gap)} from your planned price.` : "") }));

  const actions = el("div", { class: "j-actions" });
  const c = el("article", { class: `j-card ${t.status}` },
    el("div", { class: "j-head" },
      el("b", { class: "sym", text: t.ticker }), t.name && el("span", { class: "nm", text: t.name }),
      t.state && el("span", { class: `level-badge ${t.state.level}`, title: `When logged: ${LEVELS[t.state.level][1]}`, text: LEVELS[t.state.level][0] }),
      el("span", { class: `badge ${t.status}`, text: t.status === "planned" ? "plan" : t.status })),
    el("p", { class: "j-dates", text: dates }),
    planLine, outcome, checkLine,
    r.biases && r.biases.length > 0 && el("div", { class: "legend" }, ...r.biases.map((b) => el("span", {}, el("i", { style: `background: var(--hl-${b})` }), LABELS[b] || b))),
    t.reasoning && el("p", { class: "j-why", text: `"${t.reasoning}"` }),
    actions);

  if (t.status === "planned") actions.append(el("button", { type: "button", class: "link", text: "I made this trade", onclick: () => dateForm(c, t, "start") }));
  if (t.status !== "planned") actions.append(el("button", { type: "button", class: "link", text: r.status ? "Open review" : "Review", onclick: (e) => openReview(t, e.currentTarget) }));
  if (t.status === "open") actions.append(el("button", { type: "button", class: "link", text: "Close trade", onclick: () => dateForm(c, t, "close") }));
  actions.append(el("button", { type: "button", class: "link danger-text", text: "Delete", onclick: () =>
    confirmInline(actions, "Delete this trade? This can't be undone.", "Delete", async () => {
      try { await api("/delete-trade", { profile_id: profile.get(), trade_id: t.id }); refreshJournal(); }
      catch (e) { actions.append(el("span", { class: "muted", text: e.message })); }
    }) }));
  return c;
}

async function openReview(t, btn) {
  btn.disabled = true; btn.textContent = "Reviewing…";
  try {
    const r = await api("/review-logged-trade", { profile_id: profile.get(), trade_id: t.id });
    showReview(r.review, $("#j-notice"), $("#j-result"), `${t.ticker}, from your journal. `);
    refreshJournal();
  } catch (e) { btn.disabled = false; btn.textContent = "Try again"; btn.title = e.message; }
}

/** An inline form for a date: "start" a planned trade, or "close" an open one. */
function dateForm(c, t, kind) {
  c.querySelector(".date-form")?.remove();
  const short = t.direction === "short";
  const date = el("input", { type: "date", max: today(), min: kind === "close" ? t.buy_date : t.created_at.slice(0, 10), "aria-label": "Date",
    value: kind === "start" && t.buy_date && t.buy_date <= today() ? t.buy_date : "" });
  const price = el("input", { type: "number", step: "any", min: "0", placeholder: "optional", "aria-label": "Price",
    value: kind === "start" && t.plan_entry ? String(Number(t.plan_entry.toFixed(2))) : undefined });
  const f = el("form", { class: "date-form" },
    el("label", { class: "field" }, el("span", { text: kind === "start" ? (short ? "Shorted on" : "Bought on") : (short ? "Covered on" : "Sold on") }), date),
    el("label", { class: "field" }, el("span", { text: "Price" }), price),
    el("button", { type: "submit", class: "btn small", text: kind === "start" ? "Save and review" : "Close and review" }));
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!date.value) { date.classList.add("missing"); date.focus(); return; }
    const b = f.querySelector("button"); b.disabled = true;
    const price_ = price.value ? Number(price.value) : undefined;
    try {
      const r = kind === "start"
        ? await api("/start-planned-trade", { profile_id: profile.get(), trade_id: t.id, buy_date: date.value, price_paid: price_ })
        : await api("/close-trade", { profile_id: profile.get(), trade_id: t.id, sell_date: date.value, price_sold: price_ });
      showReview(r.review, $("#j-notice"), $("#j-result"), kind === "start" ? `${t.ticker} is now an open trade. ` : `${t.ticker} closed. `);
      refreshJournal();
    } catch (er) { f.append(el("span", { class: "muted", text: er.message })); b.disabled = false; }
  });
  c.append(f); date.focus();
}

export function initJournal() {
  for (const b of $$("[data-journal-refresh]")) b.addEventListener("click", refreshJournal);
}
