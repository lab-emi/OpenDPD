/**
 * The basic bundle ships without types; the app only uses `react`, `purge`
 * and `Plots.resize`, typed in components/PlotlyChart.tsx.
 */
declare module 'plotly.js-basic-dist-min' {
  const Plotly: unknown
  export default Plotly
}
