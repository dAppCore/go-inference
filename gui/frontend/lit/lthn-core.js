/* lthn-core.js — Lethean core primitives as Lit web components.
 *
 * The framework-agnostic component layer: the Wails desktop shell, Web-Awesome
 * embedded contexts, and any framework that consumes custom elements (Angular
 * pulls these in via CUSTOM_ELEMENTS_SCHEMA). Mirrors the Blade component set so a
 * screen feels identical across web and native.
 *
 * Approach (matches the desktop layer in ../desktop/):
 *   · Light DOM (createRenderRoot → this) so the global tokens + Font Awesome
 *     stylesheets apply. Load foundations/styles.css (or tokens.css) + FA6 first.
 *   · Tokens are the canonical CSS vars (--brand-*, --ink-*, --fg-*, --r-*), NOT
 *     the Tailwind --color-* names. Set data-brand / data-mode / data-platform on
 *     an ancestor to reskin.
 *   · defines are guarded, so loading this alongside ../desktop/lit-chrome.js
 *     (which ships its own chrome primitives) never double-registers.
 *
 * Elements: <lthn-button> <lthn-icon> <lthn-badge> <lthn-card> <lthn-stat>
 *           <lthn-status-dot> <lthn-state-pill> <lthn-sparkline> <lthn-divider>
 *           <lthn-input> <lthn-field> <lthn-toggle> <lthn-brand-mark>
 */
import { LitElement, html, nothing } from "https://esm.run/lit@3.1.4";
import { unsafeHTML } from "https://esm.run/lit@3.1.4/directives/unsafe-html.js";

const define = (name, cls) => { if (!customElements.get(name)) customElements.define(name, cls); };
const FA = { solid: "fas", regular: "far", light: "fal", duotone: "fad", brands: "fab" };

/* Light-DOM base: render into the element (so global tokens + Font Awesome apply)
 * and capture the original children once, since <slot> does not project without
 * a shadow root. Use ${this.slotted()} in a template where children belong. */
class LthnLight extends LitElement {
  createRenderRoot() { return this; }
  connectedCallback() { if (this.__slot === undefined) { this.__slot = this.innerHTML.trim(); this.innerHTML = ""; } super.connectedCallback(); }
  slotted() { return unsafeHTML(this.__slot || ""); }
}

/* one global rule so <lthn-brand-mark> can fall back to the --brand-name token */
if (!document.getElementById("lthn-core-style")) {
  const s = document.createElement("style");
  s.id = "lthn-core-style";
  s.textContent = ".lthn-bn:empty::before{content:var(--brand-name,'Lethean')}";
  document.head.appendChild(s);
}

/* ───────────── <lthn-icon name set size> — Font Awesome ───────────── */
class LthnIcon extends LthnLight {
  static properties = { name: {}, set: {}, size: { type: Number } };
  constructor() { super(); this.set = "solid"; this.size = 16; }
  render() {
    return html`<i class="${FA[this.set] || "fas"} fa-${this.name}" style="font-size:${this.size}px;line-height:1;display:inline-block" aria-hidden="true"></i>`;
  }
}
define("lthn-icon", LthnIcon);

/* ───────────── <lthn-button variant size icon icon-trailing disabled> ───────────── */
class LthnButton extends LthnLight {
  static properties = { variant: {}, size: {}, icon: {}, iconTrailing: { attribute: "icon-trailing" }, disabled: { type: Boolean, reflect: true }, loading: { type: Boolean, reflect: true } };
  constructor() { super(); this.variant = "primary"; this.size = "md"; }
  get styles() {
    const h = this.size === "lg" ? 48 : this.size === "sm" ? 32 : 40;
    const pad = this.size === "lg" ? 22 : this.size === "sm" ? 12 : 16;
    const r = this.size === "lg" ? "var(--r-lg)" : this.size === "sm" ? "var(--r-sm)" : "var(--r-md)";
    const fs = this.size === "lg" ? 15 : this.size === "sm" ? 13 : 14;
    const base = `display:inline-flex;align-items:center;justify-content:center;gap:8px;height:${h}px;padding:0 ${pad}px;border-radius:${r};font-weight:500;font-size:${fs}px;letter-spacing:-0.005em;border:1px solid transparent;cursor:${this.disabled ? "not-allowed" : "pointer"};opacity:${this.disabled ? 0.5 : 1};transition:background-color 120ms ease,border-color 120ms ease;font-family:inherit;`;
    const v = {
      primary: "background:var(--brand-500);color:var(--fg-0);border-color:var(--brand-400);box-shadow:inset 0 1px 0 var(--line-2),0 1px 2px rgba(0,0,0,.35);",
      secondary: "background:var(--ink-3);color:var(--fg-0);border-color:var(--line-2);",
      ghost: "background:transparent;color:var(--fg-1);",
    }[this.variant] || "";
    return base + v;
  }
  render() {
    const ip = this.size === "lg" ? 14 : this.size === "sm" ? 11 : 13;
    return html`<button ?disabled=${this.disabled || this.loading} aria-busy=${this.loading ? "true" : "false"} style=${this.styles}>
      ${this.loading ? html`<span style="width:${ip}px;height:${ip}px;border-radius:50%;border:2px solid color-mix(in oklch,var(--fg-0) 40%,transparent);border-top-color:var(--fg-0);animation:lthn-spin .7s linear infinite;display:inline-block" aria-hidden="true"></span>` : this.icon ? html`<i class="fas fa-${this.icon}" style="font-size:${ip}px" aria-hidden="true"></i>` : nothing}
      ${this.slotted()}
      ${this.iconTrailing ? html`<i class="fas fa-${this.iconTrailing}" style="font-size:${ip}px" aria-hidden="true"></i>` : nothing}
    </button>`;
  }
}
define("lthn-button", LthnButton);

/* ───────────── <lthn-badge variant icon> ───────────── */
class LthnBadge extends LthnLight {
  static properties = { variant: {}, icon: {} };
  constructor() { super(); this.variant = "neutral"; }
  render() {
    const v = {
      neutral: "background:var(--ink-3);color:var(--fg-2);border-color:var(--line-1);",
      brand: "background:color-mix(in oklch,var(--brand-500) 22%,var(--ink-2));color:var(--brand-200);border-color:color-mix(in oklch,var(--brand-500) 35%,transparent);",
      success: "background:color-mix(in oklch,var(--success-500) 22%,var(--ink-2));color:var(--success-400);border-color:color-mix(in oklch,var(--success-500) 35%,transparent);",
      warn: "background:color-mix(in oklch,var(--warning-500) 22%,var(--ink-2));color:var(--warning-400);border-color:color-mix(in oklch,var(--warning-500) 35%,transparent);",
    }[this.variant] || "";
    return html`<span style="display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 10px;border-radius:var(--r-pill);font-size:11.5px;font-weight:500;border:1px solid;${v}">
      ${this.icon ? html`<i class="fas fa-${this.icon}" style="font-size:9px" aria-hidden="true"></i>` : nothing}${this.slotted()}
    </span>`;
  }
}
define("lthn-badge", LthnBadge);

/* ───────────── <lthn-card elevated pad> ───────────── */
class LthnCard extends LthnLight {
  static properties = { elevated: { type: Boolean }, pad: { type: Number } };
  constructor() { super(); this.elevated = false; this.pad = 20; }
  render() {
    const e = this.elevated ? "border-color:var(--line-2);box-shadow:var(--shadow-2);" : "border-color:var(--line-1);";
    return html`<div style="background:var(--ink-2);border:1px solid;border-radius:var(--r-lg);padding:${this.pad}px;${e}">${this.slotted()}</div>`;
  }
}
define("lthn-card", LthnCard);

/* ───────────── <lthn-stat value label mono> ───────────── */
class LthnStat extends LthnLight {
  static properties = { value: {}, label: {}, mono: { type: Boolean } };
  render() {
    const v = this.mono
      ? "font-family:var(--font-mono);font-size:13px;font-weight:600;"
      : "font-size:22px;font-weight:600;letter-spacing:-0.02em;";
    return html`<div><div style="color:var(--fg-0);${v}">${this.value}</div><div style="font-size:12px;color:var(--fg-3);margin-top:2px">${this.label}</div></div>`;
  }
}
define("lthn-stat", LthnStat);

/* ───────────── <lthn-status-dot variant pulse> ───────────── */
class LthnStatusDot extends LthnLight {
  static properties = { variant: {}, pulse: { type: Boolean } };
  constructor() { super(); this.variant = "ok"; this.pulse = false; }
  render() {
    const c = { ok: "var(--success-400)", warn: "var(--warning-400)", err: "var(--danger-400)", idle: "var(--fg-3)", active: "var(--brand-400)" }[this.variant] || "var(--fg-3)";
    const glow = this.variant === "idle" ? "none" : `0 0 4px ${c}`;
    const anim = this.pulse ? "animation:lthn-pulse 1.4s ease-in-out infinite;" : "";
    return html`<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${c};box-shadow:${glow};${anim}"></span>`;
  }
}
define("lthn-status-dot", LthnStatusDot);

/* ───────────── <lthn-state-pill state> ───────────── */
class LthnStatePill extends LthnLight {
  static properties = { state: {} };
  constructor() { super(); this.state = "queued"; }
  render() {
    const map = {
      connected: ["color-mix(in oklch,var(--success-500) 16%,transparent)", "color-mix(in oklch,var(--success-500) 30%,transparent)", "var(--success-400)"],
      running: ["color-mix(in oklch,var(--success-500) 16%,transparent)", "color-mix(in oklch,var(--success-500) 30%,transparent)", "var(--success-400)"],
      queued: ["var(--ink-3)", "var(--line-1)", "var(--fg-2)"],
      disconnected: ["var(--ink-2)", "var(--line-1)", "var(--fg-3)"],
      preview: ["color-mix(in oklch,var(--warning-500) 16%,transparent)", "color-mix(in oklch,var(--warning-500) 30%,transparent)", "var(--warning-400)"],
      latest: ["color-mix(in oklch,var(--brand-500) 16%,transparent)", "color-mix(in oklch,var(--brand-500) 30%,transparent)", "var(--brand-300)"],
    };
    const [bg, bd, fg] = map[this.state] || map.queued;
    return html`<span style="font-family:var(--font-mono);font-size:9.5px;padding:2px 8px;border-radius:999px;background:${bg};border:1px solid ${bd};color:${fg};letter-spacing:0.06em;text-transform:uppercase;display:inline-block">${this.slotted()}</span>`;
  }
}
define("lthn-state-pill", LthnStatePill);

/* ───────────── <lthn-sparkline data color width height fill> ───────────── */
class LthnSparkline extends LthnLight {
  static properties = { data: {}, color: {}, width: { type: Number }, height: { type: Number }, fill: { type: Boolean }, max: { type: Number } };
  constructor() { super(); this.color = "var(--brand-400)"; this.width = 120; this.height = 28; this.fill = true; }
  render() {
    let s = [];
    try { s = (typeof this.data === "string" ? JSON.parse(this.data) : this.data || []).filter((n) => typeof n === "number"); } catch { s = []; }
    if (s.length < 2) s = Array.from({ length: 24 }, (_, i) => 30 + Math.sin(i * 0.7) * 10 + Math.sin(i * 1.3) * 5);
    const m = this.max || Math.max(...s) * 1.1 || 1;
    const w = this.width, h = this.height;
    const pts = s.map((v, i) => `${(i / (s.length - 1) * w).toFixed(2)} ${(h - v / m * h).toFixed(2)}`);
    const path = "M " + pts.join(" L ");
    return html`<svg viewBox="0 0 ${w} ${h}" width=${w} height=${h} style="display:block" aria-hidden="true">
      ${this.fill ? html`<path d="${path} L ${w} ${h} L 0 ${h} Z" fill=${this.color} fill-opacity="0.10"></path>` : nothing}
      <path d=${path} stroke=${this.color} stroke-width="1.5" fill="none" stroke-linejoin="round"></path>
    </svg>`;
  }
}
define("lthn-sparkline", LthnSparkline);

/* ───────────── <lthn-divider vertical> ───────────── */
class LthnDivider extends LthnLight {
  static properties = { vertical: { type: Boolean } };
  render() {
    return this.vertical
      ? html`<span style="display:inline-block;width:1px;align-self:stretch;background:var(--line-1)"></span>`
      : html`<div style="height:1px;width:100%;background:var(--line-1)"></div>`;
  }
}
define("lthn-divider", LthnDivider);

/* ───────────── <lthn-input> — themed text input ───────────── */
class LthnInput extends LthnLight {
  static properties = { type: {}, placeholder: {}, value: {} };
  constructor() { super(); this.type = "text"; }
  render() {
    return html`<input
      type=${this.type} placeholder=${this.placeholder || ""} .value=${this.value || ""}
      @input=${(e) => this.dispatchEvent(new CustomEvent("input", { detail: e.target.value }))}
      style="width:100%;height:40px;padding:0 12px;background:var(--ink-1);border:1px solid var(--line-2);border-radius:var(--r-md);color:var(--fg-0);font:inherit;box-sizing:border-box;outline:none"
      onfocus="this.style.borderColor='var(--brand-400)';this.style.background='var(--ink-2)'"
      onblur="this.style.borderColor='var(--line-2)';this.style.background='var(--ink-1)'" />`;
  }
}
define("lthn-input", LthnInput);

/* ───────────── <lthn-field label hint error> ───────────── */
class LthnField extends LthnLight {
  static properties = { label: {}, hint: {}, error: {} };
  render() {
    return html`<div>
      ${this.label ? html`<label style="display:block;font-size:12px;font-weight:500;color:var(--fg-2);margin-bottom:6px;letter-spacing:0.01em">${this.label}</label>` : nothing}
      ${this.slotted()}
      ${this.error ? html`<div style="font-size:11.5px;color:var(--danger-400);margin-top:6px">${this.error}</div>`
        : this.hint ? html`<div style="font-size:11.5px;color:var(--fg-3);margin-top:6px">${this.hint}</div>` : nothing}
    </div>`;
  }
}
define("lthn-field", LthnField);

/* ───────────── <lthn-toggle on> ───────────── */
class LthnToggle extends LthnLight {
  static properties = { on: { type: Boolean, reflect: true } };
  render() {
    const on = this.on;
    return html`<button role="switch" aria-checked=${on ? "true" : "false"}
      @click=${() => { this.on = !this.on; this.dispatchEvent(new CustomEvent("change", { detail: this.on })); }}
      style="width:38px;height:22px;border-radius:999px;border:1px solid ${on ? "var(--brand-400)" : "var(--line-2)"};background:${on ? "var(--brand-500)" : "var(--ink-3)"};position:relative;cursor:pointer;transition:background 120ms ease,border-color 120ms ease;padding:0">
      <span style="position:absolute;top:2px;left:${on ? "18px" : "2px"};width:16px;height:16px;border-radius:50%;background:var(--fg-0);transition:left 120ms ease"></span>
    </button>`;
  }
}
define("lthn-toggle", LthnToggle);

/* ───────────── <lthn-brand-mark size subdomain name> ───────────── */
class LthnBrandMark extends LthnLight {
  static properties = { size: {}, subdomain: {}, name: {} };
  constructor() { super(); this.size = "md"; }
  render() {
    const s = { sm: [14, 6], md: [17, 6], lg: [22, 8] }[this.size] || [17, 6];
    const font = s[0], r = s[1], tile = font + 8;
    return html`<div style="display:inline-flex;align-items:center;gap:10px">
      <span style="width:${tile}px;height:${tile}px;border-radius:${r}px;background:var(--brand-500);display:grid;place-items:center;box-shadow:inset 0 1px 0 var(--line-2)">
        <svg width=${font - 2} height=${font - 2} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M3 14c0-3.5 2.5-6 6.5-6.5 0-1.5 1-2.5 2.5-2.5s2.5 1 2.5 2.5l4 1.5c0 .8-.5 1.5-1.3 1.7l1 1.3-1.7.3.7 2-2-.7c-.8 1.5-2.5 2.4-4.7 2.4H8l-2 4-1-2 1-3c-1.6-.5-3-1.5-3-3.5z" fill="var(--fg-0)" opacity="0.95" />
          <circle cx="13.5" cy="9" r="0.9" fill="var(--ink-1)" />
        </svg>
      </span>
      <span style="display:flex;align-items:baseline;gap:6px">
        <span class="lthn-bn" style="font-weight:600;font-size:${font}px;letter-spacing:-0.02em;color:var(--fg-0)">${this.name || ""}</span>
        ${this.subdomain ? html`<span style="font-family:var(--font-mono);font-size:${font - 4}px;color:var(--fg-3)">/${this.subdomain}</span>` : nothing}
      </span>
    </div>`;
  }
}
define("lthn-brand-mark", LthnBrandMark);
