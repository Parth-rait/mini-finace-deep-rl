// Pages and the journal ID in the header. Each page lives in its own module.

import { $, $$, el, api, profile, onProfile, confirmInline } from "./core.js";
import { showAbout } from "./about.js";
import { initSentiment, showSentiment } from "./sentiment.js";
import { initTrades, showTrades } from "./trades.js";
import { renderStart } from "./start.js";

const PAGES = { start: null, about: showAbout, sentiment: showSentiment, trades: showTrades };

function show(page) {
  if (!(page in PAGES)) page = profile.get() ? "trades" : "start";
  for (const t of $$(".tab")) t.toggleAttribute("aria-current", t.dataset.view === page);
  for (const v of $$(".view")) v.hidden = v.id !== `view-${page}`;
  PAGES[page]?.();
  history.replaceState(null, "", `#${page}`);
  window.scrollTo({ top: 0 });
}

function renderIdChip(id) {
  const chip = $("#id-chip");
  if (!id) { chip.replaceChildren(el("button", { type: "button", class: "id-btn", text: "Start a journal", onclick: () => show("start") })); return; }
  const menu = el("div", { class: "id-menu", hidden: true },
    el("p", { class: "tag", text: "Your journal ID" }), el("p", { class: "id-mono", text: id }),
    el("button", { type: "button", class: "ghost", text: "Copy ID", onclick: async (e) => {
      try { await navigator.clipboard.writeText(id); e.currentTarget.textContent = "Copied"; } catch { e.currentTarget.textContent = "Select it above to copy"; }
    } }),
    el("button", { type: "button", class: "ghost", text: "Use another journal", onclick: () => { profile.clear(); show("start"); } }),
    el("button", { type: "button", class: "ghost danger-text", text: "Delete my journal", onclick: (e) =>
      confirmInline(menu, "Delete this journal and every trade in it? This can't be undone.", "Delete everything", async () => {
        try { await api("/delete-profile", { profile_id: id }); } catch { /* already gone */ }
        profile.clear(); show("start");
      }) }));
  const btn = el("button", { type: "button", class: "id-btn", "aria-expanded": "false", text: `${id.slice(0, 7)}…`,
    title: "Your journal", onclick: () => { menu.hidden = !menu.hidden; btn.setAttribute("aria-expanded", String(!menu.hidden)); } });
  chip.replaceChildren(btn, menu);
}

async function loadStatus() {
  try {
    const h = await api("/health");
    $("#status").textContent = `classifier ${h.bias_classifier || "none"} · prices to ${h.last_bar || "?"}`;
  } catch { $("#status").textContent = "API unreachable"; }
}

document.addEventListener("DOMContentLoaded", async () => {
  initSentiment();
  initTrades();
  for (const t of $$(".tab")) t.addEventListener("click", () => show(t.dataset.view));
  for (const a of $$("[data-go]")) a.addEventListener("click", (e) => { e.preventDefault(); show(a.dataset.go); });
  onProfile(renderIdChip);
  // a remembered ID that no longer exists (deleted, or another server) is forgotten
  const id = profile.get();
  if (id) { try { await api("/get-profile", { profile_id: id }); } catch (e) { if (e.status === 404 || e.status === 422) profile.clear(); } }
  renderIdChip(profile.get());
  renderStart($("#start-panel"), () => show("trades"));
  onProfile((pid) => { if (!pid) renderStart($("#start-panel"), () => show("trades")); });
  show((location.hash || "").slice(1).replace(/[^a-z]/g, ""));
  loadStatus();
});
