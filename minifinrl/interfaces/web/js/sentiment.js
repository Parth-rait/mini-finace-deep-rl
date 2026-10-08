// Sentiment: read any trading text for behavioural-bias signals, highlighted
// in the writer's own words, next to how accurate the classifiers are.

import { $, el, api } from "./core.js";
import { highlight, legend } from "./review-view.js";

const EXAMPLES = [
  "everyone on reddit is buying this, I can't miss it again. it literally can't go down",
  "down 40% on this one but I'm not selling at a loss, it'll come back to what I paid",
  "lost big on TSLA yesterday so I'm doubling down today to make it back",
  "bought a small position after reading the annual report; I'll cut it if margins keep falling",
];

let accuracyLoaded = false;

export function initSentiment() {
  const ex = $("#s-examples");
  ex.replaceChildren(...EXAMPLES.map((t) => el("button", { type: "button", class: "chip", text: t.length > 48 ? `${t.slice(0, 46)}…` : t,
    title: t, onclick: () => { $("#s-text").value = t; analyse(); } })));
  $("#s-form").addEventListener("submit", (e) => { e.preventDefault(); analyse(); });
}

export async function showSentiment() {
  if (accuracyLoaded) return;
  try {
    const d = await api("/research-summary");
    const c = d.classifier;
    if (!c || !c.rules || !c.aip) return;
    accuracyLoaded = true;
    const f2 = (v) => (v == null ? "-" : v.toFixed(2));
    $("#s-accuracy").replaceChildren(
      el("h2", { text: "How accurate is it?" }),
      el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {}, ...["Classifier", "Macro F1", "Cohen's kappa", "Exact match"].map((h) => el("th", { text: h })))),
        el("tbody", {}, ...[["Keyword rules", c.rules], ["Language model", c.aip]].map(([n, v]) =>
          el("tr", {}, el("td", { text: n }), el("td", { text: f2(v.macro_f1) }), el("td", { text: f2(v.macro_kappa_model) }), el("td", { text: f2(v.exact_match) })))))),
      el("p", { class: "caption", text: "Scored on 50 StockTwits posts labelled by hand (experiment E08). Kappa 0.42 is moderate agreement: useful as a prompt to reflect, not as a verdict. One labeller so far, so treat these numbers as indicative." }));
  } catch { /* the page works without the accuracy table */ }
}

async function analyse() {
  const text = $("#s-text").value.trim(), out = $("#s-result"), btn = $("#s-submit");
  if (text.length < 3) { $("#s-text").focus(); return; }
  btn.disabled = true; btn.textContent = "Reading…";
  try {
    const r = await api("/classify-biases", { text });
    const spans = r.signals.map((s) => ({ ...s, label: s.label }));
    out.replaceChildren(
      el("h2", { text: spans.length ? `${spans.length} signal${spans.length > 1 ? "s" : ""} in the text` : "No bias signals found" }),
      highlight(text, spans),
      spans.length ? legend(spans) : el("p", { class: "muted", text: "Nothing in the wording matched the six biases. That says nothing about whether the trade itself is a good idea." }),
      ...spans.map((s) => el("p", { class: "signal" }, el("b", { text: `${s.label.replace("_", " ")}: ` }), `"${s.evidence}"`)),
      el("span", { class: "tag", text: `read by ${r.classifier}. Signals describe the text, not the person.` }));
    out.hidden = false;
  } catch (e) {
    out.replaceChildren(el("p", { class: "muted", text: `Couldn't analyse that: ${e.message}` })); out.hidden = false;
  } finally { btn.disabled = false; btn.textContent = "Analyse"; }
}
