// Shared helpers. Every page talks only to this server's API, and inserts
// server and user text as text, never as HTML.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) e.append(c);
  return e;
}

export const pct = (x, d = 1) => (x == null ? "n/a" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(Math.abs(x) < 0.0005 ? 2 : d)}%`);
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const ordinal = (p) => {
  if (p < 1 || p > 99) return `${p % 1 ? p.toFixed(1) : p.toFixed(0)}th`;
  const n = Math.round(p), s = (n % 100 >= 11 && n % 100 <= 13) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th");
  return `${n}${s}`;
};
export const nice = (iso) => (iso ? new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "");
export const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const today = () => new Date().toISOString().slice(0, 10);

export const LABELS = {
  fomo: "FOMO", herding: "Herding", overconfidence: "Overconfidence", anchoring: "Anchoring",
  loss_aversion: "Loss aversion", revenge_trading: "Revenge trading",
};
export const VERDICT = {
  unusually_good: ["Better than luck explains", "good"], unusually_bad: ["Worse than luck explains", "bad"],
  within_luck_range: ["Within normal luck", "mid"],
};

export async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  let data = null;
  try { data = await r.json(); } catch { /* non-JSON error page */ }
  if (!r.ok) {
    const d = data && data.detail;
    const msg = typeof d === "string" ? d : (d && d.message) || (Array.isArray(d) ? d.map((x) => x.msg.replace(/^Value error, /, "")).join("; ") : `HTTP ${r.status}`);
    const err = new Error(msg); err.status = r.status; throw err;
  }
  return data;
}

// ---- the journal ID on this device ----------------------------------------------------------
// localStorage can be missing or blocked (private windows, strict settings), so every
// access is guarded and the site still works, just without remembering the ID.
const KEY = "tradereview.profile";
export const profile = {
  get() { try { return localStorage.getItem(KEY); } catch { return null; } },
  set(id) { try { localStorage.setItem(KEY, id); } catch { /* not remembered on this device */ } emit(); },
  clear() { try { localStorage.removeItem(KEY); } catch { /* nothing stored */ } emit(); },
};
const listeners = new Set();
export const onProfile = (fn) => listeners.add(fn);
function emit() { for (const fn of listeners) fn(profile.get()); }

// a confirm step drawn in the page (no browser dialogs)
export function confirmInline(host, question, yesLabel, onYes) {
  const box = el("div", { class: "confirm", role: "alertdialog", "aria-label": question },
    el("span", { text: question }),
    el("button", { type: "button", class: "danger", text: yesLabel, onclick: () => { box.remove(); onYes(); } }),
    el("button", { type: "button", class: "ghost", text: "Cancel", onclick: () => box.remove() }));
  host.querySelector(".confirm")?.remove();
  host.append(box);
  box.querySelector(".ghost").focus();
}

export const STATE_KINDS = {
  fear: "Fear", greed: "Greed", regret: "Regret", frustration: "Frustration",
  certainty: "Certainty", urgency: "Urgency", herd: "Herd talk",
};
export const LEVELS = {
  calm: ["Calm", "No strong emotion or pressure in the wording."],
  warm: ["Warm", "Some signs of emotion or pressure in the wording."],
  hot: ["Hot", "Strong emotion or pressure in the wording. A good moment to slow down before acting."],
};
