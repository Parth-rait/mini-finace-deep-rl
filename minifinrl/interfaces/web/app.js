// Trade Review front end. Talks only to this server's API (/search-symbols,
// /trading-days, /parse-trade, /review-trade, /research-summary, /health).
// All server and user text is inserted as text, never as HTML.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const LABELS = {
  fomo: "FOMO", herding: "Herding", overconfidence: "Overconfidence", anchoring: "Anchoring",
  loss_aversion: "Loss aversion", revenge_trading: "Revenge trading",
};
const VERDICT = {
  unusually_good: ["Better than luck explains", "good"], unusually_bad: ["Worse than luck explains", "bad"],
  within_luck_range: ["Within normal luck", "mid"],
};
// where each missing field lives in the form
const FIELD_EL = { ticker: "#f-ticker", date: "#field-date", sell_date: "#field-sell", direction: ".seg", horizon_days: "#field-sell" };

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}
const pct = (x, d = 1) => (x == null ? "n/a" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(Math.abs(x) < 0.0005 ? 2 : d)}%`);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const ordinal = (p) => {
  if (p < 1 || p > 99) return `${p % 1 ? p.toFixed(1) : p.toFixed(0)}th`;
  const n = Math.round(p), s = (n % 100 >= 11 && n % 100 <= 13) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th");
  return `${n}${s}`;
};
const nice = (iso) => (iso ? new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "");
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  let data = null;
  try { data = await r.json(); } catch { /* non-JSON error page */ }
  if (!r.ok) {
    const d = data && data.detail;
    const msg = typeof d === "string" ? d : (d && d.message) || (Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `HTTP ${r.status}`);
    const err = new Error(msg); err.status = r.status; throw err;
  }
  return data;
}

// ---- navigation ---------------------------------------------------------------------------
function show(view) {
  if (!document.getElementById(`view-${view}`)) view = "review";
  for (const t of document.querySelectorAll(".tab")) {
    if (t.dataset.view === view) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
  }
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== `view-${view}`;
  if (view === "research") loadResearch();
  history.replaceState(null, "", `#${view}`);
}

async function loadStatus() {
  try {
    const h = await api("/health");
    $("#status").textContent = `classifier ${h.bias_classifier || "none"} · prices to ${h.last_bar || "?"}`;
  } catch { $("#status").textContent = "API unreachable"; }
}

// ---- step 1: the stock -------------------------------------------------------------------
const pick = { symbol: null, name: null };
let hits = [], active = -1, searchSeq = 0;

function setPicked(symbol, name, exchange) {
  pick.symbol = symbol; pick.name = name;
  const input = $("#f-ticker");
  input.value = name ? `${symbol} · ${name}` : symbol;
  input.classList.remove("missing");
  closeList();
  const p = $("#ticker-picked");
  if (name) { p.replaceChildren(...[el("b", { text: symbol }), el("span", { text: name }), exchange && el("span", { class: "tag", text: exchange })].filter(Boolean)); p.hidden = false; }
  else p.hidden = true;
  stepState();
}

function closeList() {
  $("#ticker-list").hidden = true; $("#f-ticker").setAttribute("aria-expanded", "false"); active = -1;
}

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

function onTickerInput() {
  pick.symbol = null; pick.name = null; $("#ticker-picked").hidden = true; stepState();
  const q = $("#f-ticker").value.trim();
  if (q) search(q); else { closeList(); $("#ticker-note").hidden = true; }
}

function onTickerKey(e) {
  const open = !$("#ticker-list").hidden && hits.length;
  if (e.key === "ArrowDown" && open) { active = (active + 1) % hits.length; renderList(); e.preventDefault(); }
  else if (e.key === "ArrowUp" && open) { active = (active - 1 + hits.length) % hits.length; renderList(); e.preventDefault(); }
  else if (e.key === "Enter" && open) { const h = hits[Math.max(active, 0)]; setPicked(h.symbol, h.name, h.etf ? "ETF" : h.exchange); e.preventDefault(); }
  else if (e.key === "Escape") closeList();
}

// the symbol sent to the review: the picked one, or whatever was typed (the server checks it)
const tickerValue = () => pick.symbol || $("#f-ticker").value.trim().split(/\s|·/)[0] || "";

// ---- step 2: side and dates ----------------------------------------------------------------
const direction = () => document.querySelector('input[name="direction"]:checked').value;

function sideLabels() {
  const short = direction() === "short";
  $("#lbl-buy").textContent = short ? "Shorted on" : "Buy date";
  $("#lbl-sell").textContent = short ? "Covered on" : "Sell date";
}

function holdingState() {
  const holding = $("#f-holding").checked;
  $("#f-sell").disabled = holding; $("#field-sell").classList.toggle("off", holding);
  if (holding) $("#f-sell").value = "";
}

const countDays = debounce(async () => {
  const line = $("#days-line"), date = $("#f-date").value, sell = $("#f-sell").value, holding = $("#f-holding").checked;
  if (!date || (!sell && !holding)) { line.replaceChildren(); stepState(); return; }
  try {
    const out = await api("/trading-days", sell ? { date, sell_date: sell } : { date });
    if (out.days == null) line.replaceChildren(el("span", { class: "sub", text: out.notes.join(" ") }));
    else line.replaceChildren(...[el("b", { text: `${out.days} trading day${out.days === 1 ? "" : "s"}` }),
      el("span", { text: `, ${nice(out.entry_date)} to ${nice(out.exit_date)}` }),
      out.notes.length && el("span", { class: "sub", text: out.notes.join(" ") })].filter(Boolean));
  } catch { line.replaceChildren(); }
  stepState();
}, 200);

function stepState() {
  $("#step-stock").classList.toggle("done", Boolean(tickerValue()));
  $("#step-dates").classList.toggle("done", Boolean($("#f-date").value && ($("#f-sell").value || $("#f-holding").checked)));
  $("#step-why").classList.toggle("done", Boolean($("#f-why").value.trim()));
}

// ---- describe in words: parse, then let the person check the form --------------------------
async function fillFromWords() {
  const text = $("#f-text").value.trim(), notes = $("#fill-notes");
  if (!text) { $("#f-text").focus(); return; }
  $("#fill").disabled = true; $("#fill").textContent = "Reading…";
  try {
    const p = await api("/parse-trade", { text });
    if (p.ticker) setPicked(p.ticker, p.name, null); else { $("#f-ticker").value = ""; pick.symbol = null; }
    $("#f-date").value = p.date || "";
    $("#f-holding").checked = Boolean(p.still_holding && !p.sell_date);
    $("#f-sell").value = p.sell_date || "";
    holdingState();
    if (p.direction) document.querySelector(`input[name="direction"][value="${p.direction}"]`).checked = true;
    sideLabels();
    if (p.reasoning) $("#f-why").value = p.reasoning;
    const lines = [...p.notes];
    if (p.horizon_days && !p.sell_date && !p.still_holding) lines.push(`You said you held about ${p.horizon_days} trading days; the review will count that from the buy date unless you pick a sell date.`);
    pending.horizon = p.horizon_days && !p.sell_date && !p.still_holding ? p.horizon_days : null;
    const gaps = [!p.ticker && "the stock", !p.date && "the buy date", !(p.sell_date || p.still_holding || p.horizon_days) && "the sell date"].filter(Boolean);
    if (gaps.length) lines.push(`Still to pick: ${gaps.join(", ")}.`);
    notes.replaceChildren(...[el("span", { text: "Filled in. Check the form above, then review." }), lines.length && el("ul", {}, ...lines.map((l) => el("li", { text: l })))].filter(Boolean));
    notes.hidden = false;
    countDays(); stepState();
    $("#review-form").scrollIntoView({ block: "start" });
  } catch (e) {
    notes.replaceChildren(el("span", { text: `Couldn't read that: ${e.message}` })); notes.hidden = false;
  } finally { $("#fill").disabled = false; $("#fill").textContent = "Fill the form"; }
}
const pending = { horizon: null };

// ---- review --------------------------------------------------------------------------------
function formBody() {
  const body = { text: $("#f-why").value.trim(), direction: direction() };
  const t = tickerValue(); if (t) body.ticker = t;
  if ($("#f-date").value) body.date = $("#f-date").value;
  if ($("#f-holding").checked) body.still_holding = true;
  else if ($("#f-sell").value) body.sell_date = $("#f-sell").value;
  else if (pending.horizon) body.horizon_days = pending.horizon;
  return body;
}

function busy(on) { $("#submit").disabled = on; $("#busy").hidden = !on; }

function notice(title, lines = [], tone = "", chips = []) {
  const n = $("#notice");
  n.replaceChildren(el("h3", { text: title }));
  n.className = `notice ${tone}`;
  if (lines.length) n.append(el("ul", {}, ...lines.map((l) => el("li", { text: l }))));
  if (chips.length) {
    n.append(el("div", { class: "examples" }, el("span", { class: "examples-label", text: "Did you mean" }),
      ...chips.map((c) => el("button", { type: "button", class: "chip", text: c, onclick: () => {
        const m = c.match(/^(.*) \(([^)]+)\)$/);
        setPicked(m ? m[2] : c, m ? m[1] : null, null); submit();
      } }))));
  }
  n.hidden = false;
}

function markMissing(missing) {
  for (const [name, sel] of Object.entries(FIELD_EL)) {
    const node = $(sel);
    if (node) node.classList.toggle("missing", missing.includes(name));
  }
  if (missing.length) {
    const first = $(FIELD_EL[missing[0]]);
    if (first) { first.scrollIntoView({ block: "center" }); (first.querySelector("input") || first).focus?.(); }
  }
}

async function submit(ev) {
  if (ev) ev.preventDefault();
  closeList();
  const body = formBody();
  $("#result").hidden = true; $("#notice").hidden = true; busy(true);
  try {
    render(await api("/review-trade", body));
  } catch (e) {
    const tips = { 503: "The price data source is busy. Try again in about 30 seconds.", 429: "The language model's budget is used up for now." };
    notice("The review couldn't run.", [tips[e.status] || e.message], "bad");
  } finally { busy(false); }
}

function render(out) {
  markMissing(out.missing || []);
  const msgs = out.messages || [];
  if (out.status === "needs_input") {
    const names = { ticker: "the stock", date: "the buy date", direction: "bought or shorted", sell_date: "the sell date, or tick still holding" };
    notice("A little more is needed.", [
      `Still to pick: ${(out.missing || []).map((m) => names[m] || m).join("; ")}.`,
      ...msgs.filter((m) => !m.startsWith("Some details")),
    ], "", out.suggestions || []);
    $("#notice").scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  if (out.status !== "ok") {
    notice(out.status === "unsupported" ? "That trade can't be reviewed here." : "That trade couldn't be reviewed.", msgs, "bad", out.suggestions || []);
    $("#notice").scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  if (msgs.length) notice("Notes on this review", msgs); else $("#notice").hidden = true;
  const r = $("#result");
  r.replaceChildren(verdictCard(out), ticketCard(out), chartCard(out), reasoningCard(out), contextCard(out), explainCard(out));
  r.hidden = false;
  drawChart(out, "3d");
  (msgs.length ? $("#notice") : r).scrollIntoView({ behavior: "smooth", block: "start" });
}

function ticketCard(out) {
  const u = out.understood;
  const row = (k, v) => el("div", {}, el("dt", { text: k }), el("dd", { text: v }));
  const short = u.direction === "short";
  return el("section", { class: "card" },
    el("h2", { text: "What you did" }),
    el("p", { class: "said", text: out.what_you_did }),
    el("dl", { class: "ticket" },
      row("Stock", u.ticker + (u.name ? ` · ${u.name}` : "")),
      row("Side", short ? "Short" : "Long (bought)"),
      row(short ? "Shorted" : "Bought", nice(u.entry_date)),
      row(u.still_holding ? "Latest close" : (short ? "Covered" : "Sold"), nice(u.exit_date)),
      row("Held", `${u.horizon_days} trading days`)),
    el("span", { class: "tag", text: `trading days counted on the exchange calendar · read by ${u.parser}` }));
}

function verdictCard(out) {
  const o = out.outcome, [label, tone] = VERDICT[o.verdict];
  const card = el("section", { class: "card arch" },
    el("h2", { text: "The luck check" }),
    el("div", { class: `big ${tone}`, text: pct(o.realized_return) }),
    el("span", { class: `seal ${tone}`, text: label }),
    el("p", { class: "caption", text: `${ordinal(o.percentile)} percentile of 1,000 simulated outcomes. Market regime at entry: ${o.regime}. Plausible range (5 to 95%): ${pct(o.band_low)} to ${pct(o.band_high)}.` }));
  card.append(bandSvg(o, tone));
  return card;
}

function bandSvg(o, tone) {
  const NS = "http://www.w3.org/2000/svg", W = 300;
  const lo = Math.min(o.band_low, o.realized_return, 0) - 0.01, hi = Math.max(o.band_high, o.realized_return, 0) + 0.01;
  const x = (v) => 10 + (v - lo) / (hi - lo) * (W - 20);
  const s = document.createElementNS(NS, "svg");
  s.setAttribute("viewBox", `0 0 ${W} 56`); s.setAttribute("role", "img");
  s.setAttribute("aria-label", `Outcome ${pct(o.realized_return)} against a plausible range of ${pct(o.band_low)} to ${pct(o.band_high)}`);
  const add = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); s.append(e); return e; };
  add("line", { x1: x(0), x2: x(0), y1: 8, y2: 40, stroke: css("--rule") });
  add("rect", { x: x(o.band_low), y: 16, width: x(o.band_high) - x(o.band_low), height: 16, rx: 3, fill: css("--blue-wash") });
  add("line", { x1: x(o.synthetic_median), x2: x(o.synthetic_median), y1: 13, y2: 35, stroke: css("--blue"), "stroke-width": 1.5 });
  add("circle", { cx: x(o.realized_return), cy: 24, r: 6, fill: css(tone === "good" ? "--good" : tone === "bad" ? "--bad" : "--ink") });
  const t1 = add("text", { x: x(o.band_low), y: 52, "text-anchor": "middle" }); t1.textContent = pct(o.band_low);
  const t2 = add("text", { x: x(o.band_high), y: 52, "text-anchor": "middle" }); t2.textContent = pct(o.band_high);
  return el("div", { class: "band" }, s);
}

function chartCard(out) {
  const card = el("section", { class: "card wide" });
  const b3 = el("button", { type: "button", "aria-pressed": "true", text: "3D surface" });
  const b2 = el("button", { type: "button", "aria-pressed": "false", text: "2D range" });
  b3.onclick = () => { b3.setAttribute("aria-pressed", "true"); b2.setAttribute("aria-pressed", "false"); drawChart(out, "3d"); };
  b2.onclick = () => { b2.setAttribute("aria-pressed", "true"); b3.setAttribute("aria-pressed", "false"); drawChart(out, "2d"); };
  card.append(
    el("div", { class: "chart-head" }, el("h2", { text: "Every outcome the model considered plausible, for each day you could have held" }),
      el("div", { class: "toggle", role: "group", "aria-label": "Chart type" }, b3, b2)),
    el("div", { id: "plot", class: "plot", role: "img", "aria-label": "Distribution of simulated returns by holding day, with your trade's path" }),
    el("p", { class: "caption", text: "The surface is built from 1,000 paths simulated from your entry day's market regime. Your trade is the line: high on the surface means a common outcome, near the floor means a rare one. Drag to rotate." }));
  return card;
}

function drawChart(out, mode) {
  const box = document.getElementById("plot");
  if (!window.Plotly) { box.replaceChildren(el("p", { class: "caption", text: "The chart library didn't load (no internet?). The numbers above are complete." })); return; }
  const s = out.surface, ink = css("--ink"), muted = css("--muted"), rule = css("--rule");
  const lineColor = css(out.outcome.verdict === "unusually_bad" ? "--bad" : out.outcome.verdict === "unusually_good" ? "--good" : "--blue");
  const yPct = s.returns.map((v) => v * 100), real = s.realized.map((v) => v * 100);
  const font = { family: "IBM Plex Mono, monospace", size: 11, color: muted };
  if (mode === "3d") {
    const z = s.returns.map((_, j) => s.days.map((_, d) => s.density[d][j]));
    const zmax = Math.max(...z.flat());
    const zOnPath = s.realized.map((v, d) => {
      let j = 0; while (j < s.returns.length - 1 && Math.abs(s.returns[j + 1] - v) < Math.abs(s.returns[j] - v)) j++;
      return s.density[d][j] + zmax * 0.04;
    });
    Plotly.react(box, [
      { type: "surface", x: s.days, y: yPct, z, showscale: false, opacity: 0.92,
        colorscale: [[0, css("--card")], [0.35, css("--blue-wash")], [1, css("--blue")]],
        contours: { z: { show: true, usecolormap: true, project: { z: false }, width: 1 } },
        hovertemplate: "day %{x}<br>return %{y:.1f}%<br>likelihood %{z:.3f}<extra></extra>" },
      { type: "scatter3d", mode: "lines+markers", x: s.days, y: real, z: zOnPath, name: "your trade",
        line: { color: lineColor, width: 7 }, marker: { size: 3, color: lineColor },
        hovertemplate: "day %{x}: %{y:.1f}%<extra>your trade</extra>" },
    ], {
      margin: { l: 0, r: 0, t: 10, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", font, showlegend: false,
      scene: {
        xaxis: { title: "days held", gridcolor: rule, color: muted, dtick: Math.max(1, Math.round(s.days.length / 6)) },
        yaxis: { title: "return %", gridcolor: rule, color: muted },
        zaxis: { title: "likelihood", gridcolor: rule, color: muted, showticklabels: false },
        camera: { eye: { x: 1.35, y: -1.4, z: 0.8 }, center: { x: 0, y: 0, z: -0.1 } }, aspectratio: { x: 1.35, y: 1, z: 0.6 },
      },
    // scrollZoom off: the mouse wheel must scroll the page, not shrink the chart under the cursor
    }, { displaylogo: false, responsive: true, scrollZoom: false, modeBarButtonsToRemove: ["toImage"] });
  } else {
    const pc = (a) => a.map((v) => v * 100);
    Plotly.react(box, [
      { x: s.days, y: pc(s.p95), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x: s.days, y: pc(s.p05), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: css("--blue-wash"), name: "5 to 95% range", hovertemplate: "day %{x}: %{y:.1f}%<extra>5%</extra>" },
      { x: s.days, y: pc(s.p50), mode: "lines", line: { color: css("--blue"), width: 1.5, dash: "dot" }, name: "median" },
      { x: s.days, y: real, mode: "lines+markers", line: { color: lineColor, width: 3 }, name: "your trade" },
    ], {
      margin: { l: 50, r: 10, t: 10, b: 40 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font,
      xaxis: { title: "days held", gridcolor: rule, zeroline: false, dtick: Math.max(1, Math.round(s.days.length / 8)) },
      yaxis: { title: "return %", gridcolor: rule, zerolinecolor: muted }, legend: { orientation: "h", y: -0.25, font: { color: ink } },
    }, { displaylogo: false, responsive: true, scrollZoom: false });
  }
}

function reasoningCard(out) {
  const reasoning = out.understood.reasoning || "";
  const card = el("section", { class: "card" }, el("h2", { text: "Your reasoning, marked up" }));
  if (!reasoning) { card.append(el("p", { class: "muted", text: "No reasoning was given, so there was nothing to check for bias signals." })); return card; }
  const spans = out.biases.filter((b) => b.start != null).sort((a, b) => a.start - b.start);
  const p = el("p", { class: "reasoning" });
  let pos = 0;
  for (const b of spans) {
    if (b.start < pos) continue; // overlapping quote: keep the first mark
    p.append(reasoning.slice(pos, b.start));
    p.append(el("mark", { style: `background: var(--hl-${b.label})`, title: `${LABELS[b.label]} (${Math.round(b.confidence * 100)}% confident)`, text: reasoning.slice(b.start, b.end) }));
    pos = b.end;
  }
  p.append(reasoning.slice(pos));
  card.append(p);
  if (out.biases.length) {
    card.append(el("div", { class: "legend" }, ...out.biases.map((b) =>
      el("span", {}, el("i", { style: `background: var(--hl-${b.label})` }), `${LABELS[b.label]} `, el("span", { class: "tag", text: `${Math.round(b.confidence * 100)}%` })))));
  } else card.append(el("p", { class: "muted", text: "No bias signals found." }));
  if (out.bias_source) card.append(el("span", { class: "tag", text: `checked by ${out.bias_source}. Signals describe the text, not you.` }));
  return card;
}

function contextCard(out) {
  const rows = [["Your trade", out.outcome.realized_return], ["S&P 500 (SPY)", out.market.spy_return], ["Research stocks", out.market.universe_return]]
    .filter(([, v]) => v != null);
  const m = Math.max(0.02, ...rows.map(([, v]) => Math.abs(v)));
  return el("section", { class: "card" }, el("h2", { text: "Same days, other choices" }),
    el("div", { class: "compare" }, ...rows.map(([k, v]) => {
      const w = Math.abs(v) / m * 50, color = css(v >= 0 ? "--good" : "--bad");
      return el("div", { class: "cmp" }, el("span", { text: k }),
        el("span", { class: "track" }, el("span", { class: "zero" }),
          el("span", { class: "fill", style: `background:${color};${v >= 0 ? `left:50%` : `left:${50 - w}%`};width:${w}%` })),
        el("span", { class: "v", text: pct(v) }));
    })),
    el("p", { class: "caption", text: out.understood.direction === "short" ? "Your trade is shown as the short position's return." : "Buy-and-hold returns over your entry and exit dates." }));
}

function explainCard(out) {
  return el("section", { class: "card wide" }, el("h2", { text: "In plain words" }),
    el("p", { class: "explain", text: out.explanation }),
    el("span", { class: "tag", text: out.explanation_source === "llm" ? "written by the language model from the numbers above; every number was checked" : "built from the numbers above" }),
    el("p", { class: "caption", text: out.disclaimer }));
}

// ---- research -------------------------------------------------------------------------------
let researchLoaded = false;
async function loadResearch() {
  if (researchLoaded) return;
  const box = $("#research");
  try {
    const d = await api("/research-summary");
    researchLoaded = true;
    box.replaceChildren(...researchView(d));
  } catch (e) {
    box.replaceChildren(el("p", { class: "muted", text: `The recorded results aren't available on this instance (${e.message}).` }));
  }
}

const KIND = { buy_hold: "base", equal_weight: "base", risk_parity: "classic", min_variance: "classic", ppo: "agent", sac: "agent", td3: "agent" };
const NAME = { buy_hold: "Buy and hold", equal_weight: "Equal weight", risk_parity: "Risk parity", min_variance: "Minimum variance", ppo: "PPO", sac: "SAC", td3: "TD3" };
const f2 = (v) => (v == null ? "-" : v.toFixed(2));

function strategyTable(summary) {
  const head = el("tr", {}, ...["App", "Strategy", "Sharpe 2023-26", "Simulated", "Paths won", "Beats baseline", "Costs"].map((h) => el("th", { text: h })));
  const rows = [];
  for (const app of ["trading", "portfolio"]) {
    const models = Object.entries(summary[app] || {}).sort((a, b) => b[1].hist_sharpe - a[1].hist_sharpe);
    for (const [m, v] of models) rows.push(el("tr", {},
      el("td", { text: app }), el("td", { class: `kind-${KIND[m]}`, text: NAME[m] || m }),
      el("td", { text: f2(v.hist_sharpe) }), el("td", { text: f2(v.synth_sharpe) }),
      el("td", { text: v.wins != null ? `${v.wins}/${v.n_paths}` : "-" }), el("td", { text: v.beats_ref != null ? `${v.beats_ref}/${v.n_paths}` : "-" }),
      el("td", { text: v.cost_frac != null ? pct(v.cost_frac).replace("+", "") : "-" })));
  }
  return el("div", { class: "scroll" }, el("table", { class: "names2" }, el("thead", {}, head), el("tbody", {}, ...rows)));
}

function researchView(d) {
  const out = [
    el("h1", { text: "Do deep RL trading agents beat simple strategies? Not here." }),
    el("p", { class: "lede", text: "PPO, SAC and TD3 were trained on eight large-cap stocks (2014 to 2022) and tested on 2023 to mid-2026, on 20 simulated market histories, and across seven walk-forward years. None reliably beat buy and hold or equal weighting. Sharpe ratios are measured over the 3-month Treasury bill." }),
    el("section", { class: "card" }, el("h2", { text: "Strategies, eight stocks, three seeds each" }), strategyTable(d.main),
      el("p", { class: "caption", text: "Green: reference baselines. Blue: classical portfolio methods. Red: RL agents. Paths won counts the 20 simulated histories where a strategy came first." })),
  ];
  const wf = d.walk_forward && d.walk_forward.portfolio;
  if (wf) {
    const folds = Object.keys(wf.fold_sharpe[0]).filter((k) => k !== "strategy");
    const head = el("tr", {}, el("th", { text: "Portfolio" }), ...folds.map((f) => el("th", { text: f })));
    const rows = wf.fold_sharpe.map((r) => {
      const best = folds.map((f) => Math.max(...wf.fold_sharpe.map((x) => x[f])));
      return el("tr", {}, el("td", { class: `kind-${KIND[r.strategy]}`, text: NAME[r.strategy] || r.strategy }),
        ...folds.map((f, i) => el("td", { text: f2(r[f]), style: r[f] === best[i] ? "font-weight:700" : "" })));
    });
    const rules = el("table", {}, el("thead", {}, el("tr", {}, ...["Rule, 2020 to mid-2026", "Sharpe", "Return per year"].map((h) => el("th", { text: h })))),
      el("tbody", {}, ...wf.selection.map((r) => el("tr", {}, el("td", { text: r.rule.replace("static:", "always ").replace("follow_winner", "follow last year's winner").replace("hindsight_best", "perfect hindsight (not achievable)").replace("mix_classical", "equal thirds of the classical methods").replace(/_/g, " ") }),
        el("td", { text: f2(r.sharpe) }), el("td", { text: pct(r.cagr).replace("+", "") })))));
    out.push(el("section", { class: "card" }, el("h2", { text: "Walk-forward: retrained each year on the five before" }),
      el("div", { class: "scroll" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows))),
      el("div", { class: "scroll" }, rules),
      el("p", { class: "caption", text: "The best method changes with the year, and picking last year's winner did worse than holding one rule." })));
  }
  const c = d.classifier;
  if (c && c.rules && c.aip) {
    out.push(el("section", { class: "card" }, el("h2", { text: "Bias classifier, 50 hand-labelled posts" }),
      el("div", { class: "scroll" }, el("table", {}, el("thead", {}, el("tr", {}, ...["Classifier", "Macro F1", "Cohen's kappa", "Exact match"].map((h) => el("th", { text: h })))),
        el("tbody", {}, ...[["Keyword rules", c.rules], ["Language model", c.aip]].map(([n, v]) =>
          el("tr", {}, el("td", { text: n }), el("td", { text: f2(v.macro_f1) }), el("td", { text: f2(v.macro_kappa_model) }), el("td", { text: f2(v.exact_match) })))))),
      el("p", { class: "caption", text: "One labeller and 50 posts, so these numbers are indicative." })));
  }
  out.push(el("section", { class: "card" }, el("h2", { text: "Every recorded experiment" }),
    ...d.experiments.map((e) => el("details", { class: "exp" }, el("summary", { text: `${e.id}: ${e.title}` }), el("p", { text: e.conclusion })))));
  return out;
}

// ---- start ----------------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  const today = new Date().toISOString().slice(0, 10);
  for (const id of ["#f-date", "#f-sell"]) $(id).max = today;
  $("#review-form").addEventListener("submit", submit);
  $("#f-ticker").addEventListener("input", onTickerInput);
  $("#f-ticker").addEventListener("keydown", onTickerKey);
  $("#f-ticker").addEventListener("blur", () => setTimeout(closeList, 120));
  $("#f-ticker").addEventListener("focus", (e) => { if (pick.symbol) e.target.select(); });
  for (const r of document.querySelectorAll('input[name="direction"]')) r.addEventListener("change", sideLabels);
  $("#f-date").addEventListener("change", () => { $("#f-sell").min = $("#f-date").value; $("#field-date").classList.remove("missing"); countDays(); });
  $("#f-sell").addEventListener("change", () => { pending.horizon = null; $("#field-sell").classList.remove("missing"); countDays(); });
  $("#f-holding").addEventListener("change", () => { pending.horizon = null; holdingState(); $("#field-sell").classList.remove("missing"); countDays(); });
  $("#f-why").addEventListener("input", stepState);
  $("#fill").addEventListener("click", fillFromWords);
  for (const c of document.querySelectorAll("[data-example]")) c.addEventListener("click", () => {
    $("#f-text").value = c.dataset.example; $("#f-why").value = ""; fillFromWords();
  });
  for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => show(t.dataset.view));
  show((location.hash || "#review").slice(1).replace(/[^a-z]/g, "") || "review");
  loadStatus();
});
