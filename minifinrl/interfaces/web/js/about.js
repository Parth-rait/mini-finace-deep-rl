// About: what the project is for, the recorded research results, and how a
// review works. The research tables come from /research-summary.

import { $, el, pct, api } from "./core.js";

let loaded = false;
export async function showAbout() {
  if (loaded) return;
  const box = $("#research");
  try {
    const d = await api("/research-summary");
    loaded = true;
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

export function researchView(d) {
  const out = [
    el("h2", { class: "section-title", text: "Do deep RL trading agents beat simple strategies? Not here." }),
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

