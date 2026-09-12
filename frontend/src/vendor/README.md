# Plotly scatter build

`plotly-scatter-strict.cjs` is an unmodified Plotly 4.0.0 custom distribution,
renamed to `.cjs` so the frontend bundler recognizes its CommonJS/UMD exports.
It includes only `scatter`, `scattergl` and Plotly's calendar component. The
`--strict` build uses precompiled WebGL commands, preserving the Studio CSP.
Map modules, their attribution templates and external icon loaders are absent.
The small SVG-only package is still loaded separately for ordinary line plots.

Source: [Plotly custom-bundle instructions](https://github.com/plotly/plotly.js/blob/v4.0.0/CUSTOM_BUNDLE.md).
Upstream commit: `e020cc00ef4eb3b0b7fbf25871c2f818e73a04ce` (`v4.0.0`).
License: [MIT](LICENSE-Plotly.txt); the upstream bundle header is retained.

Rebuild in a temporary directory with Node 22 or newer (this copy was built with
Node 26.8.1 and npm 11.19.0):

```sh
git clone --depth 1 --branch v4.0.0 https://github.com/plotly/plotly.js.git
cd plotly.js
git rev-parse HEAD
npm ci --ignore-scripts
npm run custom-bundle -- --traces scatter,scattergl --strict --out opendpd-scatter
shasum -a 256 dist/plotly-opendpd-scatter.min.js
```

Expected upstream commit is the one above. The source lockfile SHA-256 is
`3691aa1e5bab7efac2bedac0e340f9245041a134a869147e0c7f661896bdee7a`.
The resulting bundle SHA-256 is
`d4f966bf3a7f2acaecc7a21734311ea984389a0ca97c62b457892f70f9a53112`.
Copy that file here as `plotly-scatter-strict.cjs` together with upstream's
`LICENSE`. Generated experiment evidence is unrelated to this runtime dependency.

After any update, run the production build, the unchanged Python offline-assets
check, plot interaction tests and a real-server WebGL/CSP browser check. Do not
replace it with the full strict distribution or allow external asset hosts.
