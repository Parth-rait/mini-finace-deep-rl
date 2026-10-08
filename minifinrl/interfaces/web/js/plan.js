// Plan: decide the trade before making it. Entry, stop, target, amount and a
// time limit, with the risk worked out as you type; then a check that adds how
// often normal luck reaches your stop and target, the state your reason is
// written in, and what to watch. It never says buy or don't buy.

import { $, $$, el, api, pct, css, debounce, profile, LEVELS } from "./core.js";
import { highlight, legend, stateView } from "./review-view.js";
import { stockPicker } from "./stock-picker.js";
import { notice } from "./review.js";
import { refreshJournal } from "./journal.js";

let stock, last = null, quote = null;
const HOLDS = [[5, "1 week"], [10, "2 weeks"], [21, "1 month"], [63, "3 months"], [126, "6 months"], [252, "1 year"]];
const num = (id) => { const v = $(id).value.trim(); return v ? Number(v) : undefined; };
const money = (x) => (x == null ? "-" : x.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: x < 100 ? 2 : 0 }));
const ratio = (x) => (x == null ? "-" : `${x.toFixed(1)} : 1`);
const share = (x) => (x == null ? "-" : `${(x * 100).toFixed(x < 0.01 ? 2 : 1)}%`);

function body() {
  return {
    ticker: stock.value(), direction: $('input[name="p-direction"]:checked').value,
    entry_price: num("#p-entry"), stop_price: num("#p-stop"), target_price: num("#p-target"),
    amount: num("#p-amount"), account_size: num("#p-account"),
    horizon_days: Number($('input[name="p-hold"]:checked').value), reasoning: $("#p-why").value.trim(),
    planned_date: $("#p-date").value || undefined,
  };
}

// ---- the arithmetic, live (the server does the same sums for the saved plan) -------------------
function preview() {
  const b = body(), out = $("#p-preview");
  const entry = b.entry_price;
  if (!entry || !b.stop_price) { out.replaceChildren(); return; }
  const long = b.direction === "long";
  const wrong = [];
  if (long ? b.stop_price >= entry : b.stop_price <= entry) wrong.push(`For a ${long ? "buy" : "short"}, the stop goes ${long ? "below" : "above"} the entry.`);
  if (b.target_price && (long ? b.target_price <= entry : b.target_price >= entry)) wrong.push(`The target goes ${long ? "above" : "below"} the entry.`);
  if (wrong.length) { out.replaceChildren(el("span", { class: "bad", text: wrong.join(" ") })); return; }
  const riskPct = Math.abs(entry - b.stop_price) / entry;
  const parts = [el("span", {}, "If the stop is hit you lose ", el("b", { text: share(riskPct) }), " of the position")];
  if (b.amount) parts.push(el("span", {}, " (", el("b", { text: money(b.amount * riskPct) }), ")"));
  if (b.target_price) {
    const rr = (Math.abs(b.target_price - entry) / entry) / riskPct;
    parts.push(el("span", {}, ". Reward to risk ", el("b", { text: ratio(rr) })));
  }
  if (b.amount && b.account_size) parts.push(el("span", {}, ". That's ", el("b", { text: share(b.amount * riskPct / b.account_size) }), " of your account"));
  parts.push(".");
  out.replaceChildren(...parts);
}

// ---- level buttons: prices from how the stock has moved, with how often luck reached them ----------
const ODDS_WORD = { entries: "fills", stops: "touched", targets: "reached" };

function chips(host, group, items, input) {
  host.replaceChildren(...items.map((l) => {
    const odds = l.prob == null ? l.note : `${ODDS_WORD[group]} in ${Math.round(l.prob * 100)}%`;
    const b = el("button", { type: "button", class: "lv", "aria-pressed": "false", title: `${l.label}: ${l.note}`,
      onclick: () => {
        input.value = String(l.price);
        for (const x of host.querySelectorAll(".lv")) x.setAttribute("aria-pressed", String(x === b));
        input.dispatchEvent(new Event("input"));
        if (group !== "targets") loadLevels();
      } },
      el("span", { class: "lv-name", text: l.label }),
      el("b", { text: l.price.toLocaleString(undefined, { maximumFractionDigits: 2 }) }),
      el("span", { class: "lv-odds", text: `${l.key === "now" ? "" : `${l.pct >= 0 ? "+" : ""}${(l.pct * 100).toFixed(1)}% · `}${odds}` }));
    if (Number(input.value) === l.price) b.setAttribute("aria-pressed", "true");
    return b;
  }));
}

let levelsSeq = 0;
const loadLevels = debounce(async () => {
  const sym = stock.value(), mine = ++levelsSeq;
  const hosts = ["#p-lv-entry", "#p-lv-stop", "#p-lv-target"].map((h) => $(h));
  if (!sym) { hosts.forEach((h) => h.replaceChildren()); $("#p-levels-info").textContent = ""; return; }
  const b = body();
  try {
    const out = await api("/plan-levels", { ticker: sym, direction: b.direction, horizon_days: b.horizon_days,
      entry_price: b.entry_price, stop_price: b.stop_price });
    if (mine !== levelsSeq) return; // a newer change is in flight
    chips(hosts[0], "entries", out.entries, $("#p-entry"));
    chips(hosts[1], "stops", out.stops, $("#p-stop"));
    chips(hosts[2], "targets", out.targets, $("#p-target"));
    $("#p-levels-info").textContent = `Typical daily move ${out.atr.toLocaleString(undefined, { maximumFractionDigits: 2 })} (${(out.atr_pct * 100).toFixed(1)}%). `
      + `Odds are how often normal luck reached each level within ${out.horizon_days} trading days, on daily closes. They describe the stock; they're not recommendations.`;
  } catch { if (mine === levelsSeq) { hosts.forEach((h) => h.replaceChildren()); $("#p-levels-info").textContent = ""; } }
}, 300);

async function prefillEntry(symbol) {
  if (!symbol) { $("#p-last").textContent = ""; quote = null; return; }
  try {
    quote = await api("/last-close", { ticker: symbol });
    $("#p-last").textContent = `Latest close ${quote.last_close.toLocaleString(undefined, { maximumFractionDigits: 2 })} on ${quote.as_of}`;
    if (!$("#p-entry").value) $("#p-entry").value = String(Number(quote.last_close.toFixed(2)));
    preview();
    loadLevels();
  } catch (e) {
    quote = null;
    // FastAPI's bare "Not Found" means the route doesn't exist: the server predates this page
    $("#p-last").textContent = e.message === "Not Found"
      ? "The server is out of date: restart it to see the latest price and the level buttons."
      : `No latest price: ${e.message}`;
  }
}

// ---- check ----------------------------------------------------------------------------------------
async function check(ev) {
  ev?.preventDefault();
  stock.close();
  if (!stock.value()) { stock.markMissing(true); return; }
  const btn = $("#p-check");
  btn.disabled = true; $("#p-busy").hidden = false; $("#p-result").hidden = true; $("#p-notice").hidden = true;
  try {
    const out = await api("/check-plan", body());
    last = out;
    if (out.status === "needs_input") { notice($("#p-notice"), "Check the prices.", out.messages, "bad"); return; }
    if (out.status !== "ok") { notice($("#p-notice"), "That plan can't be checked.", out.messages, "bad"); return; }
    if (out.messages.length) notice($("#p-notice"), "Notes", out.messages);
    render(out);
  } catch (e) {
    notice($("#p-notice"), "The check couldn't run.", [e.status === 503 ? "The price data source is busy. Try again in about 30 seconds." : e.message], "bad");
  } finally { btn.disabled = false; $("#p-busy").hidden = true; }
}

function stat(big, label, sub) {
  return el("div", { class: "stat" }, el("b", { text: big }), el("span", { text: label }), sub && el("small", { text: sub }));
}

function render(out) {
  const host = $("#p-result"), o = out.outlook, r = out.risk;
  const level = out.state ? out.state.level : null;
  const side = out.direction === "long" ? "buy" : "short";
  const head = el("section", { class: `verdict-band ${level || "none"}` },
    el("p", { class: "eyebrow", text: `${out.ticker}${out.name ? ` · ${out.name}` : ""} · ${side} at ${out.entry_price.toLocaleString()}${out.stop_price ? `, stop ${out.stop_price.toLocaleString()}` : ""}${out.target_price ? `, target ${out.target_price.toLocaleString()}` : ""} · up to ${out.horizon_days} trading days` }),
    el("h2", { class: "band-title", text: level ? `${LEVELS[level][0]}. ${LEVELS[level][1]}` : "Add your reason to see the state you're deciding in." }));

  const nudges = el("section", { class: "card" }, el("h3", { text: "Before you trade" }),
    el("ul", { class: "nudges" }, ...out.nudges.map((n) => el("li", { class: `nudge ${n.level}` },
      el("span", { class: "nudge-tag", text: { pause: "Pause", caution: "Watch", info: "Note" }[n.level] }), el("span", { text: n.text })))));

  const numbers = r && el("section", { class: "card" }, el("h3", { text: "Your plan in numbers" }),
    el("div", { class: "stats three" },
      stat(r.risk_money != null ? money(r.risk_money) : share(r.risk_pct), "lost if the stop is hit", r.risk_money != null ? `${share(r.risk_pct)} of the position` : "of the position"),
      stat(ratio(r.reward_risk), "reward to risk", r.reward_money != null ? `${money(r.reward_money)} at the target` : (out.target_price ? null : "add a target")),
      stat(share(r.account_risk_pct), "of your account at risk", r.account_risk_pct == null ? "add amount and account size" : null)));

  const luck = el("section", { class: "card" }, el("h3", { text: `What normal luck does in ${out.horizon_days} trading days` }),
    o.prob_stop != null && el("div", { class: "stats three" },
      stat(share(o.prob_stop), "of paths touch your stop"),
      stat(o.prob_target != null ? share(o.prob_target) : "-", "reach your target", o.prob_target == null ? "add a target" : null),
      stat(o.prob_target_first != null ? share(o.prob_target_first) : "-", "reach the target before the stop")),
    el("p", { class: "lead-sm", text: `Range at the end: ${pct(o.p05)} to ${pct(o.p95)} (5th to 95th percentile); ${Math.round(o.prob_loss * 100)}% of ${o.n_paths.toLocaleString()} paths lost money. Market regime today: ${o.regime}.` }),
    el("div", { class: "plot short", id: "p-fan", role: "img", "aria-label": "Range of outcomes for each day of the planned hold, with your stop and target" }),
    el("p", { class: "caption", text: `From the regime model fitted on prices up to ${o.as_of}, starting at the latest close (${o.last_price.toLocaleString()}). Checked on daily closes. Not a forecast: it shows how wide luck alone is, so you can see whether your stop sits inside ordinary noise.` }));

  const words = out.state && el("section", { class: "card" }, el("h3", { text: "Your words" }), stateView(out.state),
    out.biases.length ? el("div", { class: "stack" }, el("p", { class: "eyebrow", text: "Bias signals" }), highlight($("#p-why").value.trim(), out.biases), legend(out.biases)) : null);

  const canSave = out.stop_price && $("#p-why").value.trim().split(/\s+/).length >= 3;
  const save = el("div", { class: "actions" },
    el("button", { type: "button", class: "btn", text: "Save plan to my journal", disabled: !canSave, onclick: saveIt }),
    el("span", { class: "muted small", text: !canSave ? "To save, add a stop price and a sentence on why." :
      profile.get() ? "Your plan is frozen as it is now, and the trade is checked against it later." : "Start a journal (top right) to save plans." }));

  host.replaceChildren(head, el("div", { class: "grid-2" }, nudges, numbers || luck), numbers ? luck : null, words, save, el("p", { class: "caption", text: out.disclaimer }));
  host.hidden = false;
  drawFan(out);
  host.scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawFan(out) {
  const box = $("#p-fan"), o = out.outlook;
  if (!box) return;
  if (!window.Plotly) { box.replaceChildren(el("p", { class: "caption", text: "The chart library didn't load; the numbers above are complete." })); return; }
  const p = (a) => a.map((v) => v * 100);
  const sign = out.direction === "long" ? 1 : -1;
  const lvl = (price) => sign * (price / o.last_price - 1) * 100;
  const font = { family: "Inter, sans-serif", size: 12, color: css("--muted") };
  const lines = [];
  if (out.stop_price) lines.push({ x: [o.days[0], o.days[o.days.length - 1]], y: [lvl(out.stop_price), lvl(out.stop_price)], mode: "lines", line: { color: css("--bad"), width: 1.5, dash: "dash" }, name: "your stop", hovertemplate: "stop %{y:.1f}%<extra></extra>" });
  if (out.target_price) lines.push({ x: [o.days[0], o.days[o.days.length - 1]], y: [lvl(out.target_price), lvl(out.target_price)], mode: "lines", line: { color: css("--good"), width: 1.5, dash: "dash" }, name: "your target", hovertemplate: "target %{y:.1f}%<extra></extra>" });
  Plotly.react(box, [
    { x: o.days, y: p(o.fan_p95), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x: o.days, y: p(o.fan_p05), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: css("--accent-soft"), name: "5 to 95% of paths",
      hovertemplate: "day %{x}: %{y:.1f}%<extra>5%</extra>" },
    { x: o.days, y: p(o.fan_p50), mode: "lines", line: { color: css("--accent"), width: 2 }, name: "median path", hovertemplate: "day %{x}: %{y:.1f}%<extra>median</extra>" },
    ...lines,
  ], {
    margin: { l: 48, r: 10, t: 8, b: 36 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font,
    showlegend: lines.length > 0, legend: { orientation: "h", y: -0.3 },
    xaxis: { title: "trading days into the hold", gridcolor: css("--line"), zeroline: false },
    yaxis: { title: "return from today %", gridcolor: css("--line"), zerolinecolor: css("--muted") },
  }, { displaylogo: false, responsive: true, scrollZoom: false, displayModeBar: false });
}

async function saveIt(ev) {
  const pid = profile.get(), btn = ev.currentTarget;
  if (!pid) { notice($("#p-notice"), "Start a journal to save plans.", ["Use the button at the top right; it takes one click."]); return; }
  btn.disabled = true; btn.textContent = "Saving…";
  try {
    const b = body();
    if ($("#p-remember").checked && b.account_size) await api("/set-account-size", { profile_id: pid, account_size: b.account_size });
    await api("/save-plan", { profile_id: pid, name: stock.name() || undefined, ...b });
    btn.textContent = "Saved to your journal";
    notice($("#p-notice"), "Plan saved.", ['Find it under Journal. When you make the trade, choose "I made this trade" on it there.']);
    refreshJournal();
  } catch (e) { btn.disabled = false; btn.textContent = "Save plan to my journal"; notice($("#p-notice"), "Couldn't save the plan.", [e.message], "bad"); }
}

export async function showPlan() {
  const pid = profile.get();
  if (!pid || $("#p-account").value) return;
  try {
    const p = await api("/get-profile", { profile_id: pid });
    if (p.account_size) { $("#p-account").value = String(p.account_size); $("#p-remember").checked = true; preview(); }
  } catch { /* fine without it */ }
}

export function initPlan() {
  stock = stockPicker($("#p-stock"));
  stock.onChange((sym) => { $("#p-entry").value = ""; if (sym && stock.name()) prefillEntry(sym); else { $("#p-last").textContent = ""; } });
  stock.input.addEventListener("blur", () => { const v = stock.value(); if (v && !stock.name() && !$("#p-entry").value) prefillEntry(v); });
  $("#p-holds").replaceChildren(...HOLDS.map(([d, label], i) => el("label", {},
    el("input", { type: "radio", name: "p-hold", value: String(d), checked: i === 2 }), el("span", { text: label }))));
  $("#plan-form").addEventListener("submit", check);
  for (const id of ["#p-entry", "#p-stop", "#p-target", "#p-amount", "#p-account"]) $(id).addEventListener("input", preview);
  // typing a price clears the button that was picked; a new entry or stop moves the other levels
  for (const [id, host] of [["#p-entry", "#p-lv-entry"], ["#p-stop", "#p-lv-stop"], ["#p-target", "#p-lv-target"]]) {
    $(id).addEventListener("change", () => {
      for (const x of $(host).querySelectorAll(".lv")) x.setAttribute("aria-pressed", String(Number($(id).value) === Number(x.querySelector("b").textContent.replace(/,/g, ""))));
      if (id !== "#p-target") loadLevels();
    });
  }
  for (const r of $$('input[name="p-direction"], input[name="p-hold"]')) r.addEventListener("change", () => {
    if (r.name === "p-direction") { $("#p-stop").value = ""; $("#p-target").value = ""; }
    preview(); loadLevels(); if (last) $("#p-result").hidden = true;
  });
}
