// Sentiment: read any trading text for the writer's state (mood, emotion,
// pressure, with a heat score) and for behavioural-bias signals, each quoted in
// the writer's own words, next to how accurate the readers are.

import { $, el, api } from "./core.js";
import { highlight, legend, stateView } from "./review-view.js";

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
    const c = d.classifier, m = d.mood;
    if (!c || !c.rules || !c.aip) return;
    accuracyLoaded = true;
    const f2 = (v) => (v == null ? "-" : v.toFixed(2));
    const p0 = (v) => (v == null ? "-" : `${Math.round(v * 100)}%`);
    $("#s-accuracy").replaceChildren(
      el("h2", { text: "How accurate is it?" }),
      m && m.rules && m.aip && el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {}, ...["Mood reader", "Accuracy", "Macro F1", "Right side, tagged posts", "Right when it takes a side"].map((h) => el("th", { text: h })))),
        el("tbody", {}, ...[["Word lists", m.rules], ["Language model", m.aip]].map(([n, v]) =>
          el("tr", {}, el("td", { text: n }), el("td", { text: p0(v.accuracy) }), el("td", { text: f2(v.macro_f1) }),
            el("td", { text: p0(v.direction_accuracy) }), el("td", { text: p0(v.side_precision) })))))),
      m && m.aip && el("p", { class: "caption", text: `Mood scored on ${m.aip.texts} StockTwits posts tagged bullish or bearish by their own authors, untagged ones as neutral (experiment E09). The emotional signals have no labels yet, so their accuracy isn't measured.` }),
      el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {}, ...["Classifier", "Macro F1", "Cohen's kappa", "Exact match"].map((h) => el("th", { text: h })))),
        el("tbody", {}, ...[["Keyword rules", c.rules], ["Language model", c.aip]].map(([n, v]) =>
          el("tr", {}, el("td", { text: n }), el("td", { text: f2(v.macro_f1) }), el("td", { text: f2(v.macro_kappa_model) }), el("td", { text: f2(v.exact_match) })))))),
        el("p", { class: "caption", text: "Bias signals scored on 50 StockTwits posts labelled by hand (experiment E08). Kappa 0.42 is moderate agreement: useful as a prompt to reflect, not as a verdict. One labeller so far, so treat these numbers as indicative." }));
  } catch { /* the page works without the accuracy table */ }
}

async function analyse() {
  const text = $("#s-text").value.trim(), out = $("#s-result"), btn = $("#s-submit");
  if (text.length < 3) { $("#s-text").focus(); return; }
  btn.disabled = true; btn.textContent = "Reading…";
  const [st, bias] = await Promise.allSettled([api("/read-state", { text }), api("/classify-biases", { text })]);
  const parts = [];
  if (st.status === "fulfilled") parts.push(el("h2", { text: "The writer's state" }), stateView(st.value));
  else parts.push(el("p", { class: "muted", text: `Couldn't read the state: ${st.reason.message}` }));
  if (bias.status === "fulfilled") {
    const spans = bias.value.signals;
    parts.push(el("h2", { class: "sub-h", text: spans.length ? `Bias signals: ${spans.length}` : "No bias signals" }), highlight(text, spans),
      spans.length ? legend(spans) : el("p", { class: "muted", text: "Nothing in the wording matched the six biases. That says nothing about whether the trade itself is a good idea." }),
      el("span", { class: "tag", text: `biases read by ${bias.value.classifier}. Signals describe the text, not the person.` }));
  } else parts.push(el("p", { class: "muted", text: `Couldn't check for biases: ${bias.reason.message}` }));
  out.replaceChildren(...parts);
  out.hidden = false;
  btn.disabled = false; btn.textContent = "Analyse";
}
