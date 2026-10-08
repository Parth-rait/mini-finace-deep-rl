// A search box over every US-listed stock and ETF, with a keyboard-friendly
// dropdown. Used by the Plan and Review forms.

import { el, api, debounce } from "./core.js";

let instances = 0;

/** Turn `root` (an empty element) into a stock picker. */
export function stockPicker(root, { placeholder = "Search a company or symbol: tesla, NVDA, berkshire", label = "Stock or ETF" } = {}) {
  const n = ++instances;
  const input = el("input", { type: "text", id: `stock-${n}`, role: "combobox", "aria-autocomplete": "list", "aria-expanded": "false",
    "aria-controls": `stock-list-${n}`, placeholder, maxlength: "60", spellcheck: "false", autocomplete: "off", "aria-label": label });
  const list = el("ul", { class: "listbox", id: `stock-list-${n}`, role: "listbox", hidden: true });
  const picked = el("p", { class: "picked", hidden: true });
  const note = el("p", { class: "hint", hidden: true });
  root.replaceChildren(el("div", { class: "combo" }, input, list), picked, note);

  const state = { symbol: null, name: null };
  let hits = [], active = -1, seq = 0;
  const listeners = [];
  const changed = () => listeners.forEach((fn) => fn(api_.value()));

  function close() { list.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; }

  function render(msg) {
    list.replaceChildren(...hits.map((h, i) => {
      const li = el("li", { role: "option", id: `stock-${n}-opt-${i}`, "aria-selected": String(i === active) },
        el("span", { class: "sym", text: h.symbol }), el("span", { class: "nm", text: h.name }),
        el("span", { class: "ex", text: h.etf ? "ETF" : h.exchange }));
      li.addEventListener("mousedown", (e) => { e.preventDefault(); api_.set(h.symbol, h.name, h.etf ? "ETF" : h.exchange); });
      return li;
    }));
    if (!hits.length) list.append(el("li", { class: "empty", text: msg || "No match." }));
    list.hidden = false; input.setAttribute("aria-expanded", "true");
    input.setAttribute("aria-activedescendant", active >= 0 ? `stock-${n}-opt-${active}` : "");
    note.textContent = hits.length && msg ? msg : ""; note.hidden = !(hits.length && msg);
  }

  const search = debounce(async (q) => {
    const mine = ++seq;
    try {
      const out = await api("/search-symbols", { q, limit: 8 });
      if (mine !== seq) return; // a newer keystroke is in flight
      hits = out.results; active = hits.length ? 0 : -1; render(out.note);
    } catch (e) {
      if (mine === seq) { hits = []; render(e.status === 503 ? "The symbol list is unavailable; type the symbol and it will be checked." : e.message); }
    }
  }, 140);

  input.addEventListener("input", () => {
    state.symbol = null; state.name = null; picked.hidden = true; input.classList.remove("missing"); changed();
    const q = input.value.trim();
    if (q) search(q); else { close(); note.hidden = true; }
  });
  input.addEventListener("keydown", (e) => {
    const open = !list.hidden && hits.length;
    if (e.key === "ArrowDown" && open) { active = (active + 1) % hits.length; render(); e.preventDefault(); }
    else if (e.key === "ArrowUp" && open) { active = (active - 1 + hits.length) % hits.length; render(); e.preventDefault(); }
    else if (e.key === "Enter" && open) { const h = hits[Math.max(active, 0)]; api_.set(h.symbol, h.name, h.etf ? "ETF" : h.exchange); e.preventDefault(); }
    else if (e.key === "Escape") close();
  });
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("focus", () => { if (state.symbol) input.select(); });

  const api_ = {
    input,
    /** The picked symbol, or whatever was typed (the server checks it). */
    value: () => state.symbol || input.value.trim().split(/\s|·/)[0] || "",
    name: () => state.name,
    set(symbol, name, exchange) {
      state.symbol = symbol; state.name = name;
      input.value = name ? `${symbol} · ${name}` : symbol;
      input.classList.remove("missing");
      close();
      if (name) { picked.replaceChildren(el("b", { text: symbol }), el("span", { text: name }), exchange && el("span", { class: "tag", text: exchange })); picked.hidden = false; }
      else picked.hidden = true;
      changed();
    },
    clear() { state.symbol = null; state.name = null; input.value = ""; picked.hidden = true; changed(); },
    close,
    onChange(fn) { listeners.push(fn); },
    markMissing(on) { input.classList.toggle("missing", on); if (on) input.focus(); },
  };
  return api_;
}
