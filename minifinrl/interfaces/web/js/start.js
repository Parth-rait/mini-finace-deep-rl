// Getting a journal: start a new one (the server makes a private ID) or open
// an existing one by its ID. No account and no password: the ID is the key,
// so it's shown once, large, with a copy button and a reminder to keep it.

import { el, api, profile } from "./core.js";

/** Draw the start panel into `host`. `done(id)` runs once a journal is open. */
export function renderStart(host, done, { compact = false } = {}) {
  const msg = el("p", { class: "hint", "aria-live": "polite" });
  const idInput = el("input", { type: "text", id: "open-id", placeholder: "TR-XXXX-XXXX-XXXX-XXXX", maxlength: "40",
    spellcheck: "false", autocomplete: "off", "aria-label": "Your journal ID" });

  async function create(btn) {
    btn.disabled = true; msg.textContent = "";
    try {
      const p = await api("/create-profile", {});
      showNewId(host, p.id, done);
    } catch (e) { msg.textContent = `Couldn't start a journal: ${e.message}`; btn.disabled = false; }
  }

  async function open(ev) {
    ev.preventDefault();
    const raw = idInput.value.trim();
    if (!raw) { idInput.focus(); return; }
    msg.textContent = "";
    try {
      const p = await api("/get-profile", { profile_id: raw });
      profile.set(p.id); done(p.id);
    } catch (e) {
      msg.textContent = e.status === 404 ? "No journal has that ID. Check for typos, or start a new one." : e.message;
      idInput.classList.add("missing");
    }
  }

  host.replaceChildren(el("div", { class: compact ? "start compact" : "start" },
    el("section", { class: "start-card" },
      el("h3", { text: "New here?" }),
      el("p", { text: "Start a journal. You get a private ID, and every plan and trade you save is kept under it. No email, no password." }),
      el("button", { type: "button", class: "btn", text: "Start a new journal", onclick: (e) => create(e.currentTarget) })),
    el("form", { class: "start-card", onsubmit: open },
      el("h3", { text: "Have an ID?" }),
      el("p", { text: "Open your journal on this device." }),
      idInput,
      el("button", { type: "submit", class: "btn outline", text: "Open my journal" })),
    msg));
}

function showNewId(host, id, done) {
  const copyBtn = el("button", { type: "button", class: "btn outline small", text: "Copy ID", onclick: async () => {
    try { await navigator.clipboard.writeText(id); copyBtn.textContent = "Copied"; } catch { copyBtn.textContent = "Select and copy it"; }
  } });
  const ok = el("input", { type: "checkbox", id: "saved-id" });
  const go = el("button", { type: "button", class: "btn", text: "Continue", disabled: true,
    onclick: () => { profile.set(id); done(id); } });
  ok.addEventListener("change", () => { go.disabled = !ok.checked; });
  host.replaceChildren(el("section", { class: "start-card new-id" },
    el("h3", { text: "Your journal ID" }),
    el("p", { class: "id-big", text: id }),
    el("p", { text: "This ID is the only key to your journal. Anyone who has it can see your trades, and if you lose it the journal can't be recovered. Save it somewhere private, like a password manager or a note." }),
    el("div", { class: "row" }, copyBtn),
    el("label", { class: "check" }, ok, el("span", { text: "I've saved my ID" })),
    go));
}
