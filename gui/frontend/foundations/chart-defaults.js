/* ═══════════════════════════════════════════════════════════════════════════
   Lethean chart defaults — the graphing style pack.
   One theme object mapping design tokens → chart config. Consumed by <lthn-chart>
   (lit/lthn-charts.js) and by drop-in adapters for the charting lib your Angular
   app uses. Colours are CSS var() strings, so a chart re-themes live with
   [data-brand] / [data-mode] — no rebuild.
   ═══════════════════════════════════════════════════════════════════════════ */

export const chartTheme = {
  /* series palette — brand first, then cool→warm accents, distinct at a glance */
  palette: [
    "var(--brand-400)", "var(--info-400)", "var(--success-400)",
    "var(--gold-400)", "var(--ember-400)", "var(--brand-200)", "var(--warning-400)",
  ],
  grid: "var(--line-1)",
  axis: "var(--fg-3)",
  axisLabel: "var(--fg-4)",
  font: "var(--font-mono)",
  fontSize: 10.5,
  areaOpacity: 0.14,
  barRadius: 4,
  strokeWidth: 2,
  tooltip: { bg: "var(--ink-3)", border: "var(--line-2)", fg: "var(--fg-1)" },
};

export const seriesColor = (i) => chartTheme.palette[i % chartTheme.palette.length];

/* Resolve the var() strings to computed hex against an element's scope — handy for
   canvas-based libs (Chart.js) that can't take var() directly. Pass the host node. */
export function resolveTheme(el = document.documentElement) {
  const cs = getComputedStyle(el);
  const r = (v) => (v.startsWith("var(") ? cs.getPropertyValue(v.slice(4, -1).trim()).trim() || v : v);
  return {
    ...chartTheme,
    palette: chartTheme.palette.map(r),
    grid: r(chartTheme.grid), axis: r(chartTheme.axis), axisLabel: r(chartTheme.axisLabel),
    tooltip: { bg: r(chartTheme.tooltip.bg), border: r(chartTheme.tooltip.border), fg: r(chartTheme.tooltip.fg) },
  };
}

/* ── Adapters (examples) ──────────────────────────────────────────────────────
   ECharts:
     const t = resolveTheme(host);
     option = { color: t.palette,
       textStyle:{ fontFamily:t.font, color:t.axisLabel },
       xAxis:{ axisLine:{lineStyle:{color:t.axis}}, splitLine:{show:false} },
       yAxis:{ splitLine:{lineStyle:{color:t.grid}}, axisLine:{show:false} },
       tooltip:{ backgroundColor:t.tooltip.bg, borderColor:t.tooltip.border, textStyle:{color:t.tooltip.fg} } };

   ApexCharts:
     const t = resolveTheme(host);
     options = { colors:t.palette, grid:{borderColor:t.grid},
       xaxis:{ labels:{style:{colors:t.axisLabel, fontFamily:t.font}} },
       fill:{ type:'gradient', gradient:{opacityFrom:t.areaOpacity, opacityTo:0} } };

   ngx-charts: pass { domain: resolveTheme(host).palette } as customColors / scheme.
   ──────────────────────────────────────────────────────────────────────────── */
