// Review: a trade you have made, sold or still held. See whether the outcome
// was luck, the biases and state in your reasoning, and the market over the
// same days. Save it to your journal, or just look.

import { $, $$, el, api, nice, debounce, today, profile } from "./core.js";
import { renderReview } from "./review-view.js";
import { stockPicker } from "./stock-picker.js";
import { refreshJournal } from "./journal.js";

let stock;
const pending = { horizon: null };
const direction = () => $('input[name="r-direction"]:checked').value;
const sold = () => $('input[name="r-status"]:checked').value === "closed";

function applyStatus() {
  const short = direction() === "short";
  $("#r-lbl-buy").textContent = short ? "Shorted on" : "Bought on";
  $("#r-lbl-sell").textContent = short ? "Covered on" : "Sold on";
  $("#r-field-sell").hidden = !sold();
  if (!sold()) $("#r-sell").value = "";
  countDays();
}

const countDays = debounce(async () => {
  const line = $("#r-days"), date = $("#r-date").value, sell = $("#r-sell").value;
  if (!date || (sold() && !sell)) { line.replaceChildren(); return; }
  try {
    const out = await api("/trading-days", sell ? { date, sell_date: sell } : { date });
    if (out.days == null) line.replaceChildren(el("span", { class: "sub", text: out.notes.join(" ") }));
    else line.replaceChildren(el("b", { text: `${out.days} trading day${out.days === 1 ? "" : "s"}` }),
      el("span", { text: `, ${nice(out.entry_date)} to ${nice(out.exit_date)}${sold() ? "" : " (latest close)"}` }),
      out.notes.length && el("span", { class: "sub", text: out.notes.join(" ") }));
  } catch { line.replaceChildren(); }
}, 200);

// ---- describe in words -------------------------------------------------------------------------
async function fillFromWords() {
  const text = $("#r-text").value.trim(), notes = $("#r-fill-notes"), btn = $("#r-fill");
  if (!text) { $("#r-text").focus(); return; }
  btn.disabled = true; btn.textContent = "Reading…";
  try {
    const p = await api("/parse-trade", { text });
    if (p.ticker) stock.set(p.ticker, p.name, null); else stock.clear();
    $("#r-date").value = p.date || "";
    $("#r-sell").value = p.sell_date || "";
    $(`input[name="r-status"][value="${p.still_holding && !p.sell_date ? "open" : "closed"}"]`).checked = true;
    if (p.direction) $(`input[name="r-direction"][value="${p.direction}"]`).checked = true;
    applyStatus();
    if (p.reasoning) $("#r-why").value = p.reasoning;
    const lines = [...p.notes];
    pending.horizon = p.horizon_days && !p.sell_date && !p.still_holding ? p.horizon_days : null;
    if (pending.horizon) lines.push(`You said you held about ${p.horizon_days} trading days; that's counted from the buy date unless you pick a sell date.`);
    const gaps = [!p.ticker && "the stock", !p.date && "the buy date", !(p.sell_date || p.still_holding || p.horizon_days) && "the sell date"].filter(Boolean);
    if (gaps.length) lines.push(`Still to pick: ${gaps.join(", ")}.`);
    notes.replaceChildren(el("span", { text: "Filled in. Check the form, then review." }), lines.length && el("ul", {}, ...lines.map((l) => el("li", { text: l }))));
    notes.hidden = false;
    $("#review-form").scrollIntoView({ block: "start", behavior: "smooth" });
  } catch (e) {
    notes.replaceChildren(el("span", { text: `Couldn't read that: ${e.message}` })); notes.hidden = false;
  } finally { btn.disabled = false; btn.textContent = "Fill the form"; }
}

// ---- results and notices ------------------------------------------------------------------------
export function notice(host, title, lines = [], tone = "", chips = [], onChip = null) {
  host.replaceChildren(el("h3", { text: title }));
  host.className = `notice ${tone}`;
  if (lines.length) host.append(el("ul", {}, ...lines.map((l) => el("li", { text: l }))));
  if (chips.length && onChip) {
    host.append(el("div", { class: "examples" }, el("span", { class: "examples-label", text: "Did you mean" }),
      ...chips.map((c) => el("button", { type: "button", class: "chip", text: c, onclick: () => onChip(c) }))));
  }
  host.hidden = false;
  host.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/** Show a review outcome in `resultHost`, with notes or what's missing in `noticeHost`. */
export function showReview(out, noticeHost, resultHost, prefix = "", onChip = null) {
  const msgs = out.messages || [];
  if (out.status === "needs_input") {
    const names = { ticker: "the stock", date: "the buy date", direction: "bought or shorted", sell_date: "the sell date" };
    resultHost.hidden = true;
    notice(noticeHost, `${prefix}A little more is needed.`, [`Still to pick: ${(out.missing || []).map((m) => names[m] || m).join("; ")}.`,
      ...msgs.filter((m) => !m.startsWith("Some details"))], "", out.suggestions || [], onChip);
    return out.missing || [];
  }
  if (out.status !== "ok") {
    resultHost.hidden = true;
    notice(noticeHost, `${prefix}${out.status === "unsupported" ? "That trade can't be reviewed here." : "That trade couldn't be reviewed."}`, msgs, "bad", out.suggestions || [], onChip);
    return [];
  }
  if (msgs.length || prefix) notice(noticeHost, prefix ? prefix.trim() : "Notes on this review", msgs); else noticeHost.hidden = true;
  renderReview(resultHost, out);
  (msgs.length || prefix ? noticeHost : resultHost).scrollIntoView({ behavior: "smooth", block: "start" });
  return [];
}

const FIELDS = { ticker: () => stock.markMissing(true), date: () => mark("#r-field-date"), sell_date: () => mark("#r-field-sell"),
                 direction: () => mark("#r-side") };
function mark(sel) { const n = $(sel); n.classList.add("missing"); n.scrollIntoView({ block: "center" }); n.querySelector("input")?.focus(); }
function markMissing(missing) {
  for (const sel of ["#r-field-date", "#r-field-sell", "#r-side"]) $(sel).classList.remove("missing");
  stock.markMissing(false);
  missing.forEach((m) => FIELDS[m]?.());
}

const chipPick = (c) => { const m = c.match(/^(.*) \(([^)]+)\)$/); stock.set(m ? m[2] : c, m ? m[1] : null, null); };
const tip = (e) => ({ 503: "The price data source is busy. Try again in about 30 seconds.", 429: "The language model's budget is used up for now." }[e.status] || e.message);
const num = (sel) => { const v = $(sel).value.trim(); return v ? Number(v) : undefined; };
function busy(on, text) { $("#r-save").disabled = on; $("#r-only").disabled = on; $("#r-busy").hidden = !on; if (text) $("#r-busy-text").textContent = text; }

function checkForm() {
  const missing = [];
  if (!stock.value()) missing.push("ticker");
  if (!$("#r-date").value) missing.push("date");
  if (sold() && !$("#r-sell").value && !pending.horizon) missing.push("sell_date");
  markMissing(missing);
  return !missing.length;
}

async function reviewOnly() {
  stock.close();
  if (!checkForm()) return;
  const body = { text: $("#r-why").value.trim(), direction: direction(), ticker: stock.value(), date: $("#r-date").value };
  if (!sold()) body.still_holding = true;
  else if ($("#r-sell").value) body.sell_date = $("#r-sell").value;
  else if (pending.horizon) body.horizon_days = pending.horizon;
  $("#r-result").hidden = true; $("#r-notice").hidden = true; busy(true, "Running the luck check on 1,000 paths…");
  try { markMissing(showReview(await api("/review-trade", body), $("#r-notice"), $("#r-result"), "", chipPick)); }
  catch (e) { notice($("#r-notice"), "The review couldn't run.", [tip(e)], "bad"); }
  finally { busy(false); }
}

async function saveAndReview(ev) {
  ev?.preventDefault();
  stock.close();
  const pid = profile.get();
  if (!pid) { notice($("#r-notice"), "Start a journal to save trades.", ["Use the button at the top right. You can still review without saving."]); return; }
  if (!checkForm()) return;
  const body = { profile_id: pid, ticker: stock.value(), name: stock.name() || undefined, direction: direction(),
    buy_date: $("#r-date").value, sell_date: sold() ? $("#r-sell").value || undefined : undefined,
    price_paid: num("#r-paid"), price_sold: sold() ? num("#r-sold") : undefined, quantity: num("#r-qty"),
    reasoning: $("#r-why").value.trim() || undefined };
  $("#r-result").hidden = true; $("#r-notice").hidden = true; busy(true, "Saving, then running the luck check…");
  try {
    const t = await api("/log-trade", body);
    const r = await api("/review-logged-trade", { profile_id: pid, trade_id: t.id });
    showReview(r.review, $("#r-notice"), $("#r-result"), "Saved to your journal. ", chipPick);
    refreshJournal();
  } catch (e) { notice($("#r-notice"), "Couldn't save the trade.", [tip(e)], "bad"); }
  finally { busy(false); }
}

export function initReview() {
  stock = stockPicker($("#r-stock"));
  $("#r-date").max = today(); $("#r-sell").max = today();
  $("#review-form").addEventListener("submit", saveAndReview);
  $("#r-only").addEventListener("click", reviewOnly);
  for (const r of $$('input[name="r-direction"], input[name="r-status"]')) r.addEventListener("change", () => { pending.horizon = null; applyStatus(); });
  $("#r-date").addEventListener("change", () => { $("#r-sell").min = $("#r-date").value; $("#r-field-date").classList.remove("missing"); countDays(); });
  $("#r-sell").addEventListener("change", () => { pending.horizon = null; $("#r-field-sell").classList.remove("missing"); countDays(); });
  $("#r-fill").addEventListener("click", fillFromWords);
  for (const c of $$("[data-example]")) c.addEventListener("click", () => { $("#r-text").value = c.dataset.example; $("#r-why").value = ""; fillFromWords(); });
  applyStatus();
}
