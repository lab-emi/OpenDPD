/**
 * The strict bundle ships without types; the subset used by the app is typed
 * in components/PlotlyChart.tsx and components/plotInteractions.ts.
 */
declare module 'plotly.js-basic-dist-min' {
  const Plotly: unknown
  export default Plotly
}
declare module '@/vendor/plotly-scatter-strict.cjs' {
  const Plotly: unknown
  export default Plotly
}
