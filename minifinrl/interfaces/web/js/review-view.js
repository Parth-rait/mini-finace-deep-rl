// The result of a trade review, as cards: the luck verdict, what you did, the
// 3D outcome surface, your reasoning with bias highlights, the same days in
// the market, and the explanation. Used by the Trades page and the journal.

import { el, pct, css, ordinal, nice, LABELS, VERDICT } from "./core.js";

/** Draw a finished review (status "ok") into `host`. */
export function renderReview(host, out) {
  host.replaceChildren(verdictCard(out), ticketCard(out), chartCard(out), reasoningCard(out), contextCard(out), explainCard(out));
  host.hidden = false;
  const box = host.querySelector(".plot");
  drawChart(out, "3d", box);
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
  const box = el("div", { class: "plot", role: "img", "aria-label": "Distribution of simulated returns by holding day, with your trade's path" });
  const b3 = el("button", { type: "button", "aria-pressed": "true", text: "3D surface" });
  const b2 = el("button", { type: "button", "aria-pressed": "false", text: "2D range" });
  b3.onclick = () => { b3.setAttribute("aria-pressed", "true"); b2.setAttribute("aria-pressed", "false"); drawChart(out, "3d", box); };
  b2.onclick = () => { b2.setAttribute("aria-pressed", "true"); b3.setAttribute("aria-pressed", "false"); drawChart(out, "2d", box); };
  card.append(
    el("div", { class: "chart-head" }, el("h2", { text: "Every outcome the model considered plausible, for each day you could have held" }),
      el("div", { class: "toggle", role: "group", "aria-label": "Chart type" }, b3, b2)),
    box,
    el("p", { class: "caption", text: "The surface is built from 1,000 paths simulated from your entry day's market regime. Your trade is the line: high on the surface means a common outcome, near the floor means a rare one. Drag to rotate." }));
  return card;
}

function drawChart(out, mode, box) {
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

// The text with each signal's quoted words highlighted. `spans` carry start/end
// offsets (from the server) or are located here from their evidence.
export function highlight(text, spans) {
  const located = spans.map((b) => {
    if (b.start != null) return b;
    const i = text.toLowerCase().indexOf((b.evidence || "").toLowerCase());
    return i < 0 ? b : { ...b, start: i, end: i + b.evidence.length };
  }).filter((b) => b.start != null).sort((a, b) => a.start - b.start);
  const p = el("p", { class: "reasoning" });
  let pos = 0;
  for (const b of located) {
    if (b.start < pos) continue; // overlapping quote: keep the first mark
    p.append(text.slice(pos, b.start));
    p.append(el("mark", { style: `background: var(--hl-${b.label})`, title: `${LABELS[b.label]} (${Math.round(b.confidence * 100)}% confident)`, text: text.slice(b.start, b.end) }));
    pos = b.end;
  }
  p.append(text.slice(pos));
  return p;
}

export function legend(spans) {
  return el("div", { class: "legend" }, ...spans.map((b) =>
    el("span", {}, el("i", { style: `background: var(--hl-${b.label})` }), `${LABELS[b.label]} `, el("span", { class: "tag", text: `${Math.round(b.confidence * 100)}%` }))));
}

function reasoningCard(out) {
  const reasoning = out.understood.reasoning || "";
  const card = el("section", { class: "card" }, el("h2", { text: "Your reasoning, marked up" }));
  if (!reasoning) { card.append(el("p", { class: "muted", text: "No reasoning was given, so there was nothing to check for bias signals." })); return card; }
  card.append(highlight(reasoning, out.biases));
  if (out.biases.length) card.append(legend(out.biases));
  else card.append(el("p", { class: "muted", text: "No bias signals found." }));
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
