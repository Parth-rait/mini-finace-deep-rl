// Trades: log a trade (closed, still holding, or planned), review it, and keep
// it in your journal. The stock comes from the US symbol directory, dates from
// calendars, and trading days are counted by the server on the exchange calendar.

import { $, $$, el, api, pct, nice, ordinal, debounce, today, profile, confirmInline, VERDICT, LABELS } from "./core.js";
import { renderReview } from "./review-view.js";
import { renderStart } from "./start.js";

const FIELD_EL = { ticker: "#f-ticker", date: "#field-date", sell_date: "#field-sell", direction: "#seg-side", horizon_days: "#field-sell" };

// ---- the stock ------------------------------------------------------------------------------
const pick = { symbol: null, name: null };
let hits = [], active = -1, searchSeq = 0;

function setPicked(symbol, name, exchange) {
  pick.symbol = symbol; pick.name = name;
  const input = $("#f-ticker");
  input.value = name ? `${symbol} · ${name}` : symbol;
  input.classList.remove("missing");
  closeList();
  const p = $("#ticker-picked");
  if (name) { p.replaceChildren(el("b", { text: symbol }), el("span", { text: name }), exchange && el("span", { class: "tag", text: exchange })); p.hidden = false; }
  else p.hidden = true;
  stepState();
}

function closeList() { $("#ticker-list").hidden = true; $("#f-ticker").setAttribute("aria-expanded", "false"); active = -1; }

function renderList(note) {
  const ul = $("#ticker-list");
  ul.replaceChildren(...hits.map((h, i) => {
    const li = el("li", { role: "option", id: `opt-${i}`, "aria-selected": String(i === active) },
      el("span", { class: "sym", text: h.symbol }), el("span", { class: "nm", text: h.name }),
      el("span", { class: "ex", text: h.etf ? "ETF" : h.exchange }));
    li.addEventListener("mousedown", (e) => { e.preventDefault(); setPicked(h.symbol, h.name, h.etf ? "ETF" : h.exchange); });
    return li;
  }));
  if (!hits.length) ul.append(el("li", { class: "empty", text: note || "No match." }));
  ul.hidden = false; $("#f-ticker").setAttribute("aria-expanded", "true");
  $("#f-ticker").setAttribute("aria-activedescendant", active >= 0 ? `opt-${active}` : "");
  const n = $("#ticker-note");
  n.textContent = hits.length && note ? note : ""; n.hidden = !(hits.length && note);
}

const search = debounce(async (q) => {
  const seq = ++searchSeq;
  try {
    const out = await api("/search-symbols", { q, limit: 8 });
    if (seq !== searchSeq) return; // a newer keystroke is in flight
    hits = out.results; active = hits.length ? 0 : -1; renderList(out.note);
  } catch (e) {
    if (seq === searchSeq) { hits = []; renderList(e.status === 503 ? "The symbol list is unavailable; type the symbol and it will be checked." : e.message); }
  }
}, 140);

function onTickerKey(e) {
  const open = !$("#ticker-list").hidden && hits.length;
  if (e.key === "ArrowDown" && open) { active = (active + 1) % hits.length; renderList(); e.preventDefault(); }
  else if (e.key === "ArrowUp" && open) { active = (active - 1 + hits.length) % hits.length; renderList(); e.preventDefault(); }
  else if (e.key === "Enter" && open) { const h = hits[Math.max(active, 0)]; setPicked(h.symbol, h.name, h.etf ? "ETF" : h.exchange); e.preventDefault(); }
  else if (e.key === "Escape") closeList();
}

const tickerValue = () => pick.symbol || $("#f-ticker").value.trim().split(/\s|·/)[0] || "";

// ---- side, status and dates -----------------------------------------------------------------
const direction = () => $('input[name="direction"]:checked').value;
const status = () => $('input[name="status"]:checked').value; // closed | open | planned

function applyStatus() {
  const s = status(), short = direction() === "short";
  $("#lbl-buy").textContent = s === "planned" ? "Planned date (optional)" : (short ? "Shorted on" : "Buy date");
  $("#lbl-sell").textContent = short ? "Covered on" : "Sell date";
  $("#field-sell").hidden = s !== "closed";
  if (s !== "closed") $("#f-sell").value = "";
  $("#f-date").max = s === "planned" ? "" : today();
  $("#review-only").disabled = s === "planned";
  $("#save").textContent = s === "planned" ? "Save to my trades" : "Save and review";
  countDays();
}

const countDays = debounce(async () => {
  const line = $("#days-line"), date = $("#f-date").value, sell = $("#f-sell").value, s = status();
  if (!date || s === "planned" || (s === "closed" && !sell)) { line.replaceChildren(); stepState(); return; }
  try {
    const out = await api("/trading-days", sell ? { date, sell_date: sell } : { date });
    if (out.days == null) line.replaceChildren(el("span", { class: "sub", text: out.notes.join(" ") }));
    else line.replaceChildren(el("b", { text: `${out.days} trading day${out.days === 1 ? "" : "s"}` }),
      el("span", { text: `, ${nice(out.entry_date)} to ${nice(out.exit_date)}${s === "open" ? " (latest close)" : ""}` }),
      out.notes.length && el("span", { class: "sub", text: out.notes.join(" ") }));
  } catch { line.replaceChildren(); }
  stepState();
}, 200);

function stepState() {
  const s = status();
  $("#step-stock").classList.toggle("done", Boolean(tickerValue()));
  $("#step-dates").classList.toggle("done", s === "planned" || Boolean($("#f-date").value && (s === "open" || $("#f-sell").value)));
  $("#step-why").classList.toggle("done", Boolean($("#f-why").value.trim()));
}

// ---- describe in words: parse, then the person checks the form ------------------------------
const pending = { horizon: null };

async function fillFromWords() {
  const text = $("#f-text").value.trim(), notes = $("#fill-notes");
  if (!text) { $("#f-text").focus(); return; }
  $("#fill").disabled = true; $("#fill").textContent = "Reading…";
  try {
    const p = await api("/parse-trade", { text });
    if (p.ticker) setPicked(p.ticker, p.name, null); else { $("#f-ticker").value = ""; pick.symbol = null; }
    $("#f-date").value = p.date || "";
    $("#f-sell").value = p.sell_date || "";
    const st = p.sell_date ? "closed" : (p.still_holding ? "open" : "closed");
    $(`input[name="status"][value="${st}"]`).checked = true;
    if (p.direction) $(`input[name="direction"][value="${p.direction}"]`).checked = true;
    applyStatus();
    if (p.reasoning) $("#f-why").value = p.reasoning;
    const lines = [...p.notes];
    pending.horizon = p.horizon_days && !p.sell_date && !p.still_holding ? p.horizon_days : null;
    if (pending.horizon) lines.push(`You said you held about ${p.horizon_days} trading days; the review will count that from the buy date unless you pick a sell date.`);
    const gaps = [!p.ticker && "the stock", !p.date && "the buy date", !(p.sell_date || p.still_holding || p.horizon_days) && "the sell date"].filter(Boolean);
    if (gaps.length) lines.push(`Still to pick: ${gaps.join(", ")}.`);
    notes.replaceChildren(el("span", { text: "Filled in. Check the form above, then save or review." }), lines.length && el("ul", {}, ...lines.map((l) => el("li", { text: l }))));
    notes.hidden = false;
    countDays(); stepState();
    $("#trade-form").scrollIntoView({ block: "start", behavior: "smooth" });
  } catch (e) {
    notes.replaceChildren(el("span", { text: `Couldn't read that: ${e.message}` })); notes.hidden = false;
  } finally { $("#fill").disabled = false; $("#fill").textContent = "Fill the form"; }
}

// ---- notices --------------------------------------------------------------------------------
function notice(title, lines = [], tone = "", chips = []) {
  const n = $("#notice");
  n.replaceChildren(el("h3", { text: title }));
  n.className = `notice ${tone}`;
  if (lines.length) n.append(el("ul", {}, ...lines.map((l) => el("li", { text: l }))));
  if (chips.length) {
    n.append(el("div", { class: "examples" }, el("span", { class: "examples-label", text: "Did you mean" }),
      ...chips.map((c) => el("button", { type: "button", class: "chip", text: c, onclick: () => {
        const m = c.match(/^(.*) \(([^)]+)\)$/);
        setPicked(m ? m[2] : c, m ? m[1] : null, null);
      } }))));
  }
  n.hidden = false;
  n.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function markMissing(missing) {
  for (const [name, sel] of Object.entries(FIELD_EL)) $(sel)?.classList.toggle("missing", missing.includes(name));
  if (missing.length) { const first = $(FIELD_EL[missing[0]]); first?.scrollIntoView({ block: "center" }); (first?.querySelector("input") || first)?.focus?.(); }
}

/** Show a review outcome: the cards when it ran, otherwise what's needed. */
function showReview(out, prefix = "") {
  markMissing(out.missing || []);
  const msgs = out.messages || [];
  const result = $("#result");
  if (out.status === "needs_input") {
    const names = { ticker: "the stock", date: "the buy date", direction: "bought or shorted", sell_date: "the sell date" };
    result.hidden = true;
    notice(`${prefix}A little more is needed.`, [`Still to pick: ${(out.missing || []).map((m) => names[m] || m).join("; ")}.`,
      ...msgs.filter((m) => !m.startsWith("Some details"))], "", out.suggestions || []);
    return;
  }
  if (out.status !== "ok") {
    result.hidden = true;
    notice(`${prefix}${out.status === "unsupported" ? "That trade can't be reviewed here." : "That trade couldn't be reviewed."}`, msgs, "bad", out.suggestions || []);
    return;
  }
  if (msgs.length || prefix) notice(prefix ? prefix.trim() : "Notes on this review", msgs); else $("#notice").hidden = true;
  renderReview(result, out);
  (msgs.length || prefix ? $("#notice") : result).scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- review only, or save and review ----------------------------------------------------------
function checkForm() {
  const missing = [];
  if (!tickerValue()) missing.push("ticker");
  if (status() !== "planned" && !$("#f-date").value) missing.push("date");
  if (status() === "closed" && !$("#f-sell").value && !pending.horizon) missing.push("sell_date");
  markMissing(missing);
  return missing.length === 0;
}

const num = (sel) => { const v = $(sel).value.trim(); return v ? Number(v) : undefined; };
function busy(on, text) { $("#save").disabled = on; $("#review-only").disabled = on || status() === "planned"; $("#busy").hidden = !on; if (text) $("#busy-text").textContent = text; }

async function reviewOnly() {
  closeList();
  if (!checkForm()) return;
  const body = { text: $("#f-why").value.trim(), direction: direction(), ticker: tickerValue(), date: $("#f-date").value };
  if (status() === "open") body.still_holding = true;
  else if ($("#f-sell").value) body.sell_date = $("#f-sell").value;
  else if (pending.horizon) body.horizon_days = pending.horizon;
  $("#result").hidden = true; $("#notice").hidden = true; busy(true, "Running the luck check on 1,000 paths…");
  try { showReview(await api("/review-trade", body)); }
  catch (e) { notice("The review couldn't run.", [tip(e)], "bad"); }
  finally { busy(false); }
}

async function saveAndReview(ev) {
  ev?.preventDefault();
  closeList();
  const pid = profile.get();
  if (!pid) { showGate("Start or open a journal to save trades. You can still review without saving."); return; }
  if (!checkForm()) return;
  const s = status();
  const body = { profile_id: pid, ticker: tickerValue(), name: pick.name || undefined, direction: direction(),
    planned: s === "planned", buy_date: $("#f-date").value || undefined, sell_date: s === "closed" ? $("#f-sell").value || undefined : undefined,
    price_paid: num("#f-paid"), price_sold: s === "closed" ? num("#f-sold") : undefined, quantity: num("#f-qty"),
    reasoning: $("#f-why").value.trim() || undefined };
  $("#result").hidden = true; $("#notice").hidden = true; busy(true, s === "planned" ? "Saving…" : "Saving, then running the luck check…");
  try {
    const t = await api("/log-trade", body);
    if (s === "planned") { notice("Saved to your trades as planned.", ["When you make the trade, log it with its buy date to have it reviewed."]); }
    else { const r = await api("/review-logged-trade", { profile_id: pid, trade_id: t.id }); showReview(r.review, "Saved to your trades. "); }
    loadJournal();
  } catch (e) { notice("Couldn't save the trade.", [tip(e)], "bad"); }
  finally { busy(false); }
}

const tip = (e) => ({ 503: "The price data source is busy. Try again in about 30 seconds.", 429: "The language model's budget is used up for now." }[e.status] || e.message);

// ---- the journal ----------------------------------------------------------------------------
function showGate(message) {
  const gate = $("#trades-gate");
  gate.hidden = false;
  gate.replaceChildren(el("p", { class: "gate-msg", text: message }), el("div", { id: "gate-start" }));
  renderStart($("#gate-start"), () => { gate.hidden = true; loadJournal(); });
  gate.scrollIntoView({ behavior: "smooth", block: "start" });
}

export async function loadJournal() {
  const pid = profile.get(), list = $("#j-list"), summary = $("#j-summary");
  $("#journal").hidden = !pid;
  if (!pid) return;
  try {
    const d = await api("/list-trades", { profile_id: pid });
    const s = d.summary;
    summary.textContent = s.count ? `${s.count} trade${s.count > 1 ? "s" : ""}: ${s.closed} closed, ${s.open} open, ${s.planned} planned` : "";
    list.replaceChildren(...(d.trades.length ? d.trades.map(tradeCard) : [el("p", { class: "muted", text: "No trades yet. Log one above and it will appear here." })]));
  } catch (e) {
    if (e.status === 404) { profile.clear(); showGate("That journal no longer exists on this server. Start a new one or open another."); return; }
    list.replaceChildren(el("p", { class: "muted", text: `Couldn't load your trades: ${e.message}` }));
  }
}

function tradeCard(t) {
  const short = t.direction === "short";
  const r = t.result;
  const dates = t.status === "planned" ? (t.buy_date ? `Planned for ${nice(t.buy_date)}` : "Planned")
    : `${short ? "Shorted" : "Bought"} ${nice(t.buy_date)} · ${t.sell_date ? `${short ? "covered" : "sold"} ${nice(t.sell_date)}` : "still holding"}`;
  let outcome = el("span", { class: "muted", text: t.status === "planned" ? "No outcome yet" : "Not reviewed yet" });
  if (r && r.status === "ok") {
    const [label, tone] = VERDICT[r.verdict];
    outcome = el("span", { class: "outcome" }, el("b", { class: `ret ${r.your_return >= 0 ? "up" : "down"}`, text: pct(r.your_return) }),
      el("span", { class: `seal small ${tone}`, text: label }), el("span", { class: "tag", text: `${ordinal(r.percentile)} pct · ${r.days}d${t.status === "open" ? ` to ${nice(r.exit_date)}` : ""}` }));
  } else if (r) outcome = el("span", { class: "muted", text: r.message || "The review couldn't run." });
  const actions = el("div", { class: "j-actions" });
  const card = el("article", { class: `j-card status-${t.status}` },
    el("div", { class: "j-head" }, el("b", { class: "sym", text: t.ticker }), t.name && el("span", { class: "nm", text: t.name }),
      el("span", { class: `badge ${t.status}`, text: t.status })),
    el("p", { class: "j-dates", text: dates }),
    outcome,
    r && r.biases && r.biases.length && el("div", { class: "legend" }, ...r.biases.map((b) => el("span", {}, el("i", { style: `background: var(--hl-${b})` }), LABELS[b] || b))),
    t.reasoning && el("p", { class: "j-why", text: `"${t.reasoning}"` }),
    actions);
  if (t.status !== "planned") actions.append(el("button", { type: "button", class: "ghost", text: r ? "Open review" : "Review", onclick: (e) => openReview(t, e.currentTarget) }));
  if (t.status === "open") actions.append(el("button", { type: "button", class: "ghost", text: "Close trade", onclick: () => closeForm(card, t) }));
  actions.append(el("button", { type: "button", class: "ghost danger-text", text: "Delete", onclick: () =>
    confirmInline(actions, "Delete this trade? This can't be undone.", "Delete", async () => {
      try { await api("/delete-trade", { profile_id: profile.get(), trade_id: t.id }); loadJournal(); }
      catch (e) { actions.append(el("span", { class: "muted", text: e.message })); }
    }) }));
  return card;
}

async function openReview(t, btn) {
  btn.disabled = true; btn.textContent = "Reviewing…";
  try {
    const r = await api("/review-logged-trade", { profile_id: profile.get(), trade_id: t.id });
    showReview(r.review, `${t.ticker}, from your journal. `);
    loadJournal();
  } catch (e) { btn.textContent = "Try again"; btn.disabled = false; btn.title = e.message; }
}

function closeForm(card, t) {
  card.querySelector(".close-form")?.remove();
  const date = el("input", { type: "date", min: t.buy_date, max: today(), "aria-label": "Sell date" });
  const price = el("input", { type: "number", step: "any", min: "0", placeholder: "price (optional)", "aria-label": "Price sold" });
  const f = el("form", { class: "close-form" }, el("label", { class: "field" }, el("span", { text: t.direction === "short" ? "Covered on" : "Sold on" }), date),
    el("label", { class: "field" }, el("span", { text: "Price" }), price),
    el("button", { type: "submit", class: "primary small", text: "Close and review" }));
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!date.value) { date.classList.add("missing"); date.focus(); return; }
    f.querySelector("button").disabled = true;
    try {
      const r = await api("/close-trade", { profile_id: profile.get(), trade_id: t.id, sell_date: date.value, price_sold: price.value ? Number(price.value) : undefined });
      showReview(r.review, `${t.ticker} closed. `); loadJournal();
    } catch (er) { f.append(el("span", { class: "muted", text: er.message })); f.querySelector("button").disabled = false; }
  });
  card.append(f); date.focus();
}

// ---- wiring -----------------------------------------------------------------------------------
export function initTrades() {
  $("#f-date").max = today(); $("#f-sell").max = today();
  $("#trade-form").addEventListener("submit", saveAndReview);
  $("#review-only").addEventListener("click", reviewOnly);
  $("#f-ticker").addEventListener("input", () => {
    pick.symbol = null; pick.name = null; $("#ticker-picked").hidden = true; stepState();
    const q = $("#f-ticker").value.trim();
    if (q) search(q); else { closeList(); $("#ticker-note").hidden = true; }
  });
  $("#f-ticker").addEventListener("keydown", onTickerKey);
  $("#f-ticker").addEventListener("blur", () => setTimeout(closeList, 120));
  $("#f-ticker").addEventListener("focus", (e) => { if (pick.symbol) e.target.select(); });
  for (const r of $$('input[name="direction"], input[name="status"]')) r.addEventListener("change", () => { pending.horizon = null; applyStatus(); });
  $("#f-date").addEventListener("change", () => { $("#f-sell").min = $("#f-date").value; $("#field-date").classList.remove("missing"); countDays(); });
  $("#f-sell").addEventListener("change", () => { pending.horizon = null; $("#field-sell").classList.remove("missing"); countDays(); });
  $("#f-why").addEventListener("input", stepState);
  $("#fill").addEventListener("click", fillFromWords);
  for (const c of $$("[data-example]")) c.addEventListener("click", () => { $("#f-text").value = c.dataset.example; $("#f-why").value = ""; fillFromWords(); });
  applyStatus();
}

export function showTrades() {
  if (!profile.get()) showGate("Start or open a journal to keep your trades. You can also review a trade without saving it.");
  else $("#trades-gate").hidden = true;
  loadJournal();
}
