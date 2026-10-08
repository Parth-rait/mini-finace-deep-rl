// Pages and the journal ID in the header. Each page lives in its own module.

import { $, $$, el, api, profile, onProfile, confirmInline } from "./core.js";
import { showAbout } from "./about.js";
import { initSentiment, showSentiment } from "./sentiment.js";
import { initPlan, showPlan } from "./plan.js";
import { initReview } from "./review.js";
import { initJournal, showJournal } from "./journal.js";
import { renderStart } from "./start.js";

const PAGES = { home: null, plan: showPlan, review: null, journal: showJournal, sentiment: showSentiment, about: showAbout };

function show(page) {
  if (!(page in PAGES)) page = "home";
  for (const t of $$("[data-view]")) t.toggleAttribute("aria-current", t.dataset.view === page);
  for (const v of $$(".view")) v.hidden = v.id !== `view-${page}`;
  PAGES[page]?.();
  history.replaceState(null, "", page === "home" ? location.pathname : `#${page}`);
  window.scrollTo({ top: 0 });
  closeMenus();
}

function closeMenus() { for (const m of $$(".id-menu")) m.hidden = true; }

function renderIdChip(id) {
  const chip = $("#id-chip");
  const menu = el("div", { class: "id-menu", hidden: true });
  const btn = el("button", { type: "button", class: id ? "id-btn" : "btn small", "aria-expanded": "false",
    text: id ? `${id.slice(0, 7)}…` : "Start a journal", title: id ? "Your journal" : "Start or open a journal",
    onclick: (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; btn.setAttribute("aria-expanded", String(!menu.hidden)); } });
  menu.addEventListener("click", (e) => e.stopPropagation());
  if (id) {
    menu.append(
      el("p", { class: "eyebrow", text: "Your journal ID" }), el("p", { class: "id-mono", text: id }),
      el("button", { type: "button", class: "link", text: "Copy ID", onclick: async (e) => {
        try { await navigator.clipboard.writeText(id); e.currentTarget.textContent = "Copied"; } catch { e.currentTarget.textContent = "Select it above to copy"; }
      } }),
      el("button", { type: "button", class: "link", text: "Open my journal", onclick: () => show("journal") }),
      el("button", { type: "button", class: "link", text: "Use another journal", onclick: () => { profile.clear(); show("home"); } }),
      el("button", { type: "button", class: "link danger-text", text: "Delete my journal", onclick: () =>
        confirmInline(menu, "Delete this journal and every trade in it? This can't be undone.", "Delete everything", async () => {
          try { await api("/delete-profile", { profile_id: id }); } catch { /* already gone */ }
          profile.clear(); show("home");
        }) }));
  } else {
    const host = el("div", {});
    menu.append(host);
    renderStart(host, () => { closeMenus(); show("journal"); }, { compact: true });
  }
  chip.replaceChildren(btn, menu);
}

function renderHomeJournal(id) {
  const host = $("#home-journal");
  if (id) host.replaceChildren(el("div", { class: "row-between" },
    el("div", {}, el("p", { class: "eyebrow", text: "Your journal" }), el("p", { class: "id-mono", text: id })),
    el("button", { type: "button", class: "btn", text: "Open my journal", onclick: () => show("journal") })));
  else renderStart(host, () => show("journal"));
}

async function loadStatus() {
  try {
    const h = await api("/health");
    $("#status").textContent = `prices to ${h.last_bar || "?"} · ${h.bias_classifier || "no classifier"}`;
  } catch { $("#status").textContent = "API unreachable"; }
}

document.addEventListener("DOMContentLoaded", async () => {
  initPlan(); initReview(); initJournal(); initSentiment();
  for (const t of $$("[data-view]")) t.addEventListener("click", (e) => { e.preventDefault(); show(t.dataset.view); });
  document.addEventListener("click", closeMenus);
  // a remembered ID that no longer exists (deleted, or another server) is forgotten
  const id = profile.get();
  if (id) { try { await api("/get-profile", { profile_id: id }); } catch (e) { if (e.status === 404 || e.status === 422) profile.clear(); } }
  onProfile((pid) => { renderIdChip(pid); renderHomeJournal(pid); });
  renderIdChip(profile.get()); renderHomeJournal(profile.get());
  show((location.hash || "").slice(1).replace(/[^a-z]/g, "") || "home");
  loadStatus();
});
