/* lthn-datatable.js — <lthn-datatable>, the design system's data-grid default.
 *
 * Dense, sortable, optionally selectable table on the tokens. columns + rows are
 * JSON attributes; cell rendering is by column type (text · num · mono · status ·
 * badge). Sorting + selection are handled internally and emit events.
 *
 *   <lthn-datatable selectable page-size="8" loading  (or)  empty-label="No models yet"
 *     columns='[{"key":"name","label":"Model"},{"key":"rate","label":"tok/s","type":"num"},
 *               {"key":"status","label":"State","type":"status"}]'
 *     rows='[{"name":"llama-3.1-70b","rate":34.2,"status":"running"}, …]'></lthn-datatable>
 *
 * States: `loading` → skeleton rows · no rows → the `empty-label` message.
 * A11y: <th scope=col> with aria-sort; sort headers are real <button>s (keyboard +
 *   focus-visible ring); checkboxes are labelled; the grid sets aria-busy when loading.
 * Events: `sort` {key,dir} · `selection` {rows} · `rowclick` {row}.
 */
import { LitElement, html, nothing } from "https://esm.run/lit@3.1.4";

const define = (n, c) => { if (!customElements.get(n)) customElements.define(n, c); };
const STATUS = {
  running: ["success", "Running"], connected: ["success", "Connected"], active: ["success", "Active"],
  queued: ["neutral", "Queued"], idle: ["neutral", "Idle"], disconnected: ["neutral", "Off"],
  warning: ["warn", "Warning"], due: ["warn", "Payment due"], preview: ["warn", "Preview"],
  error: ["danger", "Error"], stalled: ["danger", "Stalled"], failed: ["danger", "Failed"],
};
const PILL = {
  success: ["var(--success-500)", "var(--success-400)"], warn: ["var(--warning-500)", "var(--warning-400)"],
  danger: ["var(--danger-500)", "var(--danger-400)"], neutral: [null, "var(--fg-3)"],
};

class LthnDatatable extends LitElement {
  static properties = {
    columns: {}, rows: {}, selectable: { type: Boolean }, loading: { type: Boolean }, emptyLabel: { attribute: "empty-label" },
    sortKey: { state: true }, sortDir: { state: true }, page: { state: true }, pageSize: { type: Number, attribute: "page-size" }, _sel: { state: true },
  };
  constructor() { super(); this.selectable = false; this.loading = false; this.emptyLabel = "No rows to show."; this.sortDir = "asc"; this.sortKey = ""; this.page = 0; this.pageSize = 0; this._sel = new Set(); }
  createRenderRoot() { return this; }

  get cols() { try { return typeof this.columns === "string" ? JSON.parse(this.columns) : (this.columns || []); } catch { return []; } }
  get data() { try { return typeof this.rows === "string" ? JSON.parse(this.rows) : (this.rows || []); } catch { return []; } }

  get sorted() {
    const d = this.data.map((r, i) => ({ r, i }));
    if (this.sortKey) {
      const k = this.sortKey, dir = this.sortDir === "asc" ? 1 : -1;
      d.sort((a, b) => { const x = a.r[k], y = b.r[k]; return (typeof x === "number" && typeof y === "number") ? (x - y) * dir : String(x ?? "").localeCompare(String(y ?? "")) * dir; });
    }
    return d;
  }
  get paged() { const s = this.sorted; if (!this.pageSize) return s; const start = this.page * this.pageSize; return s.slice(start, start + this.pageSize); }

  _sort(c) {
    if (c.sortable === false) return;
    if (this.sortKey === c.key) this.sortDir = this.sortDir === "asc" ? "desc" : "asc"; else { this.sortKey = c.key; this.sortDir = "asc"; }
    this.dispatchEvent(new CustomEvent("sort", { detail: { key: this.sortKey, dir: this.sortDir } }));
  }
  _toggle(i) { const s = new Set(this._sel); s.has(i) ? s.delete(i) : s.add(i); this._sel = s; this._emitSel(); }
  _toggleAll() { const all = this.data.map((_, i) => i); this._sel = this._sel.size === all.length ? new Set() : new Set(all); this._emitSel(); }
  _emitSel() { this.dispatchEvent(new CustomEvent("selection", { detail: { rows: [...this._sel].map((i) => this.data[i]) } })); }

  _cell(c, row) {
    const v = row[c.key];
    if (c.type === "status") {
      const [tone, label] = STATUS[v] || ["neutral", v];
      const [bg, fg] = PILL[tone] || PILL.neutral;
      const bgc = bg ? `color-mix(in oklch, ${bg} 16%, transparent)` : "var(--ink-3)";
      const bdc = bg ? `color-mix(in oklch, ${bg} 30%, transparent)` : "var(--line-1)";
      return html`<span style="font-family:var(--font-mono);font-size:9.5px;padding:2px 8px;border-radius:999px;background:${bgc};border:1px solid ${bdc};color:${fg};letter-spacing:.06em;text-transform:uppercase">${label}</span>`;
    }
    if (c.type === "badge") return html`<span style="display:inline-flex;height:20px;align-items:center;padding:0 8px;border-radius:999px;background:color-mix(in oklch,var(--brand-500) 20%,var(--ink-2));color:var(--brand-200);font-size:11px">${v}</span>`;
    return v;
  }

  render() {
    const cols = this.cols, total = this.data.length;
    const span = cols.length + (this.selectable ? 1 : 0);
    const numAlign = (c) => (c.align || (c.type === "num" || c.type === "mono" ? "right" : "left"));
    const mono = (c) => (c.type === "num" || c.type === "mono");
    const th = "padding:9px 14px;font-family:var(--font-mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--fg-3);border-bottom:1px solid var(--line-1);white-space:nowrap;user-select:none";
    const td = "padding:9px 14px;border-bottom:1px solid var(--line-1);color:var(--fg-1);font-size:13px";
    const pages = this.pageSize ? Math.ceil(total / this.pageSize) : 1;
    const skelRows = this.pageSize || 5;

    return html`<div role="region" aria-label="Data table" aria-busy=${this.loading ? "true" : "false"} style="border:1px solid var(--line-1);border-radius:var(--r-lg,12px);overflow:hidden;background:var(--ink-2)">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr>
          ${this.selectable ? html`<th scope="col" style="${th};text-align:left;width:36px"><input type="checkbox" aria-label="Select all rows" ?disabled=${this.loading} .checked=${this._sel.size === total && total > 0} @change=${() => this._toggleAll()} style="accent-color:var(--brand-500)"></th>` : nothing}
          ${cols.map((c) => {
            const sortable = c.sortable !== false;
            const active = this.sortKey === c.key;
            const ariaSort = !sortable ? undefined : active ? (this.sortDir === "asc" ? "ascending" : "descending") : "none";
            const caret = active ? html`<span aria-hidden="true" style="color:var(--brand-300);margin-left:5px">${this.sortDir === "asc" ? "\u2191" : "\u2193"}</span>` : nothing;
            return html`<th scope="col" aria-sort=${ariaSort ?? nothing} style="${th};text-align:${numAlign(c)}">
              ${sortable
                ? html`<button @click=${() => this._sort(c)} aria-label="Sort by ${c.label}" style="all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:2px;font:inherit;letter-spacing:inherit;text-transform:inherit;color:inherit">${c.label}${caret}</button>`
                : html`${c.label}`}
            </th>`;
          })}
        </tr></thead>
        <tbody>
          ${this.loading
            ? Array.from({ length: skelRows }).map(() => html`<tr>
                ${this.selectable ? html`<td style="${td}"><span class="lthn-skeleton" style="display:block;width:16px;height:16px"></span></td>` : nothing}
                ${cols.map((c, ci) => html`<td style="${td};text-align:${numAlign(c)}"><span class="lthn-skeleton" style="display:inline-block;height:11px;width:${ci === 0 ? 60 : 34}%"></span></td>`)}
              </tr>`)
            : total === 0
              ? html`<tr><td colspan=${span} style="padding:34px 14px;text-align:center;color:var(--fg-3);font-size:13px">${this.emptyLabel}</td></tr>`
              : this.paged.map(({ r, i }) => html`<tr @click=${() => this.dispatchEvent(new CustomEvent("rowclick", { detail: { row: r } }))}
                  style="background:${this._sel.has(i) ? "color-mix(in oklch,var(--brand-500) 10%,transparent)" : "transparent"}"
                  @mouseenter=${(e) => { if (!this._sel.has(i)) e.currentTarget.style.background = "var(--ink-3)"; }}
                  @mouseleave=${(e) => { if (!this._sel.has(i)) e.currentTarget.style.background = "transparent"; }}>
                  ${this.selectable ? html`<td style="${td};width:36px" @click=${(e) => e.stopPropagation()}><input type="checkbox" aria-label="Select row" .checked=${this._sel.has(i)} @change=${() => this._toggle(i)} style="accent-color:var(--brand-500)"></td>` : nothing}
                  ${cols.map((c) => html`<td style="${td};text-align:${numAlign(c)};${mono(c) ? "font-family:var(--font-mono);font-variant-numeric:tabular-nums;color:var(--fg-2)" : ""}">${this._cell(c, r)}</td>`)}
                </tr>`)}
        </tbody>
      </table>
      <div style="display:flex;align-items:center;justify-content:space-between;padding:9px 14px;font-size:12px;color:var(--fg-3)">
        <span>${this.loading ? "Loading…" : `${this._sel.size ? `${this._sel.size} selected · ` : ""}${total} rows`}</span>
        ${!this.loading && pages > 1 ? html`<span style="display:flex;align-items:center;gap:10px;font-family:var(--font-mono)">
          <button ?disabled=${this.page === 0} @click=${() => this.page = Math.max(0, this.page - 1)} aria-label="Previous page" style="background:none;border:0;color:${this.page === 0 ? "var(--fg-4)" : "var(--fg-1)"};cursor:pointer">‹</button>
          ${this.page + 1} / ${pages}
          <button ?disabled=${this.page >= pages - 1} @click=${() => this.page = Math.min(pages - 1, this.page + 1)} aria-label="Next page" style="background:none;border:0;color:${this.page >= pages - 1 ? "var(--fg-4)" : "var(--fg-1)"};cursor:pointer">›</button>
        </span>` : nothing}
      </div>
    </div>`;
  }
}
define("lthn-datatable", LthnDatatable);
