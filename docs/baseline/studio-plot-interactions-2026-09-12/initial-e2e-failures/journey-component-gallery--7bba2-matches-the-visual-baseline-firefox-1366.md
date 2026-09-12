# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: journey.spec.ts >> component gallery >> renders real-size charts quickly and matches the visual baseline
- Location: e2e/journey.spec.ts:171:3

# Error details

```
Error: expect(received).toBeLessThan(expected)

Expected: < 2000
Received:   2114
```

# Page snapshot

```yaml
- generic [ref=e3]:
  - link "Skip to main content" [ref=e4] [cursor=pointer]:
    - /url: "#main"
  - generic [ref=e6]:
    - generic [ref=e10]:
      - paragraph [ref=e11]: OpenDPD
      - text: STUDIO
    - navigation "primary" [ref=e12]:
      - link "Home" [ref=e13] [cursor=pointer]:
        - /url: /
      - link "Datasets" [ref=e19] [cursor=pointer]:
        - /url: /datasets
      - link "Experiments" [ref=e25] [cursor=pointer]:
        - /url: /experiments
      - link "Results" [ref=e31] [cursor=pointer]:
        - /url: /results
      - link "Robustness" [ref=e38] [cursor=pointer]:
        - /url: /robustness
      - link "Settings" [ref=e44] [cursor=pointer]:
        - /url: /settings
      - link "Component gallery (mock)" [ref=e50] [cursor=pointer]:
        - /url: /gallery
    - generic [ref=e56]:
      - text: Local workspace
      - paragraph [ref=e57]: v2.2.0.dev0
  - banner [ref=e58]:
    - generic [ref=e59]:
      - paragraph [ref=e60]: Component gallery (mock)
      - paragraph [ref=e62]: opendpd workspace
      - button [ref=e63] [cursor=pointer]
      - button [ref=e67] [cursor=pointer]
      - generic [ref=e71]: Nothing running
      - button "Language" [ref=e73] [cursor=pointer]:
        - generic [ref=e75]: English
  - main [ref=e79]:
    - generic [ref=e80]:
      - heading "Component gallery" [level=1] [ref=e81]
      - alert [ref=e82]:
        - generic [ref=e86]: Every domain component rendered on fixed mock fixtures generated from the contract examples. Nothing here is a computed result.
      - alert [ref=e87]:
        - generic [ref=e91]: "Performance probe: PSD with 2560 points × 3 traces and an I/Q window of 20000 samples rendered in 2114 ms."
      - region "StatusChip" [ref=e92]:
        - separator [ref=e93]:
          - heading "StatusChip" [level=2] [ref=e95]
        - generic [ref=e96]:
          - generic [ref=e97]: Queued
          - generic [ref=e101]: Running
          - generic [ref=e105]: Stopping…
          - generic [ref=e109]: Cancelled
          - generic [ref=e113]: Succeeded
          - generic [ref=e117]: Failed
          - generic [ref=e121]: Interrupted
          - 'generic "Log disconnected: no heartbeat for over a minute" [ref=e125]': Running
      - region "EvidenceBadge" [ref=e129]:
        - separator [ref=e130]:
          - heading "EvidenceBadge" [level=2] [ref=e132]
        - generic [ref=e133]:
          - generic "How well the model reproduces the measured PA output. Not a DPD result." [ref=e134]: PA model
          - generic "DPD evaluated against the trained PA model, not against a real PA." [ref=e136]: DPD · surrogate
          - generic "DPD scored on captures of a physical PA. The captures and conditions are operator-provided and not independently verified." [ref=e138]: DPD · measured
          - generic "Mock data for interface development; not a computed result and never exportable." [ref=e140]: MOCK · PA model
      - region "MetricCard" [ref=e142]:
        - separator [ref=e143]:
          - heading "MetricCard" [level=2] [ref=e145]
        - generic [ref=e146]:
          - region "NMSE" [ref=e148]:
            - generic [ref=e149]:
              - heading "NMSE" [level=3] [ref=e151]
              - paragraph [ref=e152]: "-16.09 dB"
              - generic [ref=e153]: lower is better
          - region "EVM" [ref=e158]:
            - generic [ref=e159]:
              - heading "EVM" [level=3] [ref=e161]
              - paragraph [ref=e162]: "-16.73 dB"
              - generic [ref=e163]: lower is better
          - region "ACLR_L" [ref=e168]:
            - generic [ref=e169]:
              - heading "ACLR_L" [level=3] [ref=e171]
              - paragraph [ref=e172]: "-29.79 dBc"
              - generic [ref=e173]: lower is better
          - region "ACLR_R" [ref=e178]:
            - generic [ref=e179]:
              - heading "ACLR_R" [level=3] [ref=e181]
              - paragraph [ref=e182]: "-26.35 dBc"
              - generic [ref=e183]: lower is better
          - region "ACLR_AVG" [ref=e188]:
            - generic [ref=e189]:
              - heading "ACLR_AVG" [level=3] [ref=e191]
              - paragraph [ref=e192]: "-28.07 dBc"
              - generic [ref=e193]: lower is better
          - region "EVM" [ref=e198]:
            - generic [ref=e199]:
              - heading "EVM" [level=3] [ref=e201]
              - paragraph [ref=e202]: not applicable
              - generic [ref=e203]: lower is better
              - paragraph [ref=e207]: sample_rate_hz, bandwidth_hz, n_sub_ch and nperseg are unknown for dataset my-pa-capture
          - region "ACLR_L" [ref=e209]:
            - generic [ref=e210]:
              - heading "ACLR_L" [level=3] [ref=e212]
              - paragraph [ref=e213]: not applicable
              - generic [ref=e214]: lower is better
              - paragraph [ref=e218]: sample_rate_hz, bandwidth_hz, n_sub_ch and nperseg are unknown for dataset my-pa-capture
          - region "ACLR_R" [ref=e220]:
            - generic [ref=e221]:
              - heading "ACLR_R" [level=3] [ref=e223]
              - paragraph [ref=e224]: not applicable
              - generic [ref=e225]: lower is better
              - paragraph [ref=e229]: sample_rate_hz, bandwidth_hz, n_sub_ch and nperseg are unknown for dataset my-pa-capture
          - region "ACLR_AVG" [ref=e231]:
            - generic [ref=e232]:
              - heading "ACLR_AVG" [level=3] [ref=e234]
              - paragraph [ref=e235]: not applicable
              - generic [ref=e236]: lower is better
              - paragraph [ref=e240]: sample_rate_hz, bandwidth_hz, n_sub_ch and nperseg are unknown for dataset my-pa-capture
      - region "SpectrumPlot / IQPreview" [ref=e241]:
        - separator [ref=e242]:
          - heading "SpectrumPlot / IQPreview" [level=2] [ref=e244]
        - generic [ref=e245]:
          - generic [ref=e247]:
            - generic [ref=e248]:
              - heading "Power spectral density" [level=3] [ref=e249]
              - button "Plot controls" [ref=e250] [cursor=pointer]
              - 'button "Enlarge chart: Power spectral density" [ref=e253] [cursor=pointer]'
            - generic [ref=e256]: "Two fingers: pan · Pinch or Ctrl/Cmd + wheel: zoom · Double-click: fit"
            - figure "Power spectral density" [ref=e257]:
              - generic [ref=e259]:
                - img:
                  - generic:
                    - generic:
                      - generic:
                        - generic: −400
                        - generic: −200
                        - generic: "0"
                        - generic: "200"
                      - generic:
                        - generic: −100
                        - generic: −80
                        - generic: −60
                        - generic: −40
                        - generic: −20
                - img:
                  - generic:
                    - generic [ref=e276]:
                      - generic [ref=e277]:
                        - text: input
                        - generic [ref=e281] [cursor=pointer]
                      - generic [ref=e282]:
                        - text: PA output
                        - generic [ref=e286] [cursor=pointer]
                      - generic [ref=e287]:
                        - text: with DPD
                        - generic [ref=e291] [cursor=pointer]
                    - generic: Frequency (MHz)
                    - generic: PSD (dB/Hz)
                - toolbar [ref=e292]:
                  - generic [ref=e293]:
                    - button "Download plot as a PNG" [ref=e294] [cursor=pointer]
                    - button "Share chart..." [ref=e297] [cursor=pointer]
                  - generic [ref=e300]:
                    - button "Zoom" [ref=e301] [cursor=pointer]
                    - button "Pan" [ref=e304] [cursor=pointer]
                  - generic [ref=e307]:
                    - button "Zoom in" [ref=e308] [cursor=pointer]
                    - button "Zoom out" [ref=e311] [cursor=pointer]
                    - button "Autoscale" [ref=e314] [cursor=pointer]
                    - button "Reset axes" [ref=e317] [cursor=pointer]
          - generic [ref=e321]:
            - generic [ref=e322]:
              - heading "I/Q time window" [level=3] [ref=e323]
              - button "Plot controls" [ref=e324] [cursor=pointer]
              - 'button "Enlarge chart: I/Q time window" [ref=e327] [cursor=pointer]'
            - generic [ref=e330]: "Two fingers: pan · Pinch or Ctrl/Cmd + wheel: zoom · Double-click: fit"
            - figure "I/Q time window" [ref=e331]:
              - generic [ref=e333]:
                - img:
                  - generic:
                    - generic:
                      - generic:
                        - generic: "0"
                        - generic: 5k
                        - generic: 10k
                        - generic: 15k
                        - generic: 20k
                      - generic:
                        - generic: −1
                        - generic: −0.5
                        - generic: "0"
                        - generic: "0.5"
                        - generic: "1"
                - img:
                  - generic:
                    - generic [ref=e350]:
                      - generic [ref=e351]:
                        - text: input I
                        - generic [ref=e355] [cursor=pointer]
                      - generic [ref=e356]:
                        - text: input Q
                        - generic [ref=e360] [cursor=pointer]
                      - generic [ref=e361]:
                        - text: output I
                        - generic [ref=e365] [cursor=pointer]
                      - generic [ref=e366]:
                        - text: output Q
                        - generic [ref=e370] [cursor=pointer]
                    - generic: Sample
                    - generic: Amplitude
                - toolbar [ref=e371]:
                  - generic [ref=e372]:
                    - button "Download plot as a PNG" [ref=e373] [cursor=pointer]
                    - button "Share chart..." [ref=e376] [cursor=pointer]
                  - generic [ref=e379]:
                    - button "Zoom" [ref=e380] [cursor=pointer]
                    - button "Pan" [ref=e383] [cursor=pointer]
                  - generic [ref=e386]:
                    - button "Zoom in" [ref=e387] [cursor=pointer]
                    - button "Zoom out" [ref=e390] [cursor=pointer]
                    - button "Autoscale" [ref=e393] [cursor=pointer]
                    - button "Reset axes" [ref=e396] [cursor=pointer]
          - generic [ref=e400]:
            - generic [ref=e401]:
              - heading "NMSE per epoch" [level=3] [ref=e402]
              - button "Plot controls" [ref=e403] [cursor=pointer]
              - 'button "Enlarge chart: NMSE per epoch" [ref=e406] [cursor=pointer]'
            - generic [ref=e409]: "Two fingers: pan · Pinch or Ctrl/Cmd + wheel: zoom · Double-click: fit"
            - figure "NMSE per epoch" [ref=e410]:
              - generic [ref=e412]:
                - img:
                  - generic:
                    - generic:
                      - generic:
                        - generic: −1
                        - generic: −0.5
                        - generic: "0"
                        - generic: "0.5"
                        - generic: "1"
                      - generic:
                        - generic: −17
                        - generic: −16.5
                        - generic: −16
                        - generic: −15.5
                - img:
                  - generic:
                    - generic [ref=e430]:
                      - text: val NMSE
                      - generic [ref=e437] [cursor=pointer]
                    - generic: Epoch
                    - generic: NMSE
                - toolbar [ref=e438]:
                  - generic [ref=e439]:
                    - button "Download plot as a PNG" [ref=e440] [cursor=pointer]
                    - button "Share chart..." [ref=e443] [cursor=pointer]
                  - generic [ref=e446]:
                    - button "Zoom" [ref=e447] [cursor=pointer]
                    - button "Pan" [ref=e450] [cursor=pointer]
                  - generic [ref=e453]:
                    - button "Zoom in" [ref=e454] [cursor=pointer]
                    - button "Zoom out" [ref=e457] [cursor=pointer]
                    - button "Autoscale" [ref=e460] [cursor=pointer]
                    - button "Reset axes" [ref=e463] [cursor=pointer]
      - region "DiagnosticItem" [ref=e466]:
        - separator [ref=e467]:
          - heading "DiagnosticItem" [level=2] [ref=e469]
        - generic [ref=e470]:
          - article "Sample rate and bandwidth unknown" [ref=e471]:
            - generic [ref=e472]:
              - heading "Sample rate and bandwidth unknown" [level=4] [ref=e475]
              - generic [ref=e476]: error
              - generic [ref=e478]: blocks evaluation
              - code [ref=e481]: missing_metadata
            - paragraph [ref=e482]: ACLR and spectral EVM need sample_rate_hz, bandwidth_hz, n_sub_ch and nperseg.
            - paragraph [ref=e483]:
              - strong [ref=e484]: "Evidence:"
              - generic [ref=e485]: missing=sample_rate_hz,bandwidth_hz,n_sub_ch,nperseg
            - paragraph [ref=e486]:
              - strong [ref=e487]: "Suggestion:"
              - text: Enter the signal parameters in the dataset manifest, then re-run the Doctor.
          - article "Output lags input by about 3 samples" [ref=e488]:
            - generic [ref=e489]:
              - heading "Output lags input by about 3 samples" [level=4] [ref=e492]
              - generic [ref=e493]: info
              - generic [ref=e495]: confidence 0.85
              - code [ref=e498]: delay_estimate
            - paragraph [ref=e499]: Cross-correlation peak at +3 samples (fractional refinement +0.12).
            - paragraph [ref=e500]:
              - strong [ref=e501]: "Evidence:"
              - generic [ref=e502]: fractional_delay=0.12
              - generic [ref=e503]: integer_delay=3
              - generic [ref=e504]: peak_ratio=0.93
            - paragraph [ref=e505]:
              - strong [ref=e506]: "Suggestion:"
              - text: Review and apply the alignment as a new preprocessing version.
          - article "Flat-top samples detected" [ref=e507]:
            - generic [ref=e508]:
              - heading "Flat-top samples detected" [level=4] [ref=e512]
              - generic [ref=e513]: warning
              - generic [ref=e515]: confidence 0.60
              - code [ref=e518]: possible_clipping
            - paragraph [ref=e519]: 0.4% of output samples sit within 0.1% of the maximum amplitude.
            - paragraph [ref=e520]:
              - strong [ref=e521]: "Evidence:"
              - generic [ref=e522]: fraction_at_max=0.004
              - generic [ref=e523]: max_amplitude=0.998
            - paragraph [ref=e524]:
              - strong [ref=e525]: "Suggestion:"
              - text: Natural PA compression can look similar; check the capture range.
      - region "RunTimeline" [ref=e526]:
        - separator [ref=e527]:
          - heading "RunTimeline" [level=2] [ref=e529]
        - region "Timeline" [ref=e530]:
          - heading "Timeline" [level=2] [ref=e531]
          - list [ref=e532]:
            - listitem [ref=e533]:
              - generic [ref=e534]: 10:00:02 AM
              - generic [ref=e535]: Running
            - listitem [ref=e539]:
              - generic [ref=e540]: 1 × heartbeat · last 10:00:32 AM
      - region "ConfigDiff" [ref=e541]:
        - separator [ref=e542]:
          - heading "ConfigDiff" [level=2] [ref=e544]
        - table "Configuration differences" [ref=e545]:
          - rowgroup [ref=e546]:
            - row [ref=e547]:
              - columnheader "Field" [ref=e548]
              - columnheader "smoke" [ref=e549]
              - columnheader "research" [ref=e550]
          - rowgroup [ref=e551]:
            - row [ref=e552]:
              - cell [ref=e553]:
                - code [ref=e554]: model.parameters.hidden_size
              - cell [ref=e555]:
                - code [ref=e556]: "23"
              - cell [ref=e557]:
                - code [ref=e558]: "32"
            - row [ref=e559]:
              - cell [ref=e560]:
                - code [ref=e561]: training.epochs
              - cell [ref=e562]:
                - code [ref=e563]: "3"
              - cell [ref=e564]:
                - code [ref=e565]: "300"
            - row [ref=e566]:
              - cell [ref=e567]:
                - code [ref=e568]: training.learning_rate
              - cell [ref=e569]:
                - code [ref=e570]: "0.005"
              - cell [ref=e571]:
                - code [ref=e572]: "0.001"
      - region "States" [ref=e573]:
        - separator [ref=e574]:
          - heading "States" [level=2] [ref=e576]
        - generic [ref=e577]:
          - status [ref=e578]:
            - progressbar [aria-hidden] [ref=e579]
            - paragraph [ref=e582]: Loading…
          - generic [ref=e583]:
            - paragraph [ref=e584]: Nothing here yet
            - paragraph [ref=e585]: No experiments yet.
          - alert [ref=e586]:
            - generic [ref=e590]:
              - generic [ref=e591]: Something went wrong
              - text: PA surrogate checkpoint for run-pa-0007 is not registered.
            - button "Retry" [ref=e593] [cursor=pointer]
          - status [ref=e594]:
            - generic [ref=e598]:
              - generic [ref=e599]: Live updates disconnected
              - text: Reconnecting… last update 10:00:32 AM. The run keeps going on the server; refresh to fetch the latest snapshot.
            - button "Refresh now" [ref=e601] [cursor=pointer]
      - region "ResultView (mock)" [ref=e602]:
        - separator [ref=e603]:
          - heading "ResultView (mock)" [level=2] [ref=e605]
        - generic [ref=e606]:
          - generic [ref=e607]:
            - heading "res-pa-0001" [level=1] [ref=e608]
            - generic "Mock data for interface development; not a computed result and never exportable." [ref=e609]: MOCK · PA model
            - generic [ref=e611]: legacy-opendpd-v1 v1
            - paragraph [ref=e613]:
              - text: "Run:"
              - link "run-pa-0001" [ref=e614] [cursor=pointer]:
                - /url: /runs/run-pa-0001
          - paragraph [ref=e615]: "Protocol: legacy-opendpd-v1 v1 · 3 segments × 2560 samples · selected epoch 1 · cpu · float32"
          - generic [ref=e616]:
            - region "NMSE" [ref=e618]:
              - generic [ref=e619]:
                - heading "NMSE" [level=3] [ref=e621]
                - paragraph [ref=e622]: "-16.09 dB"
                - generic [ref=e623]: lower is better
            - region "EVM" [ref=e628]:
              - generic [ref=e629]:
                - heading "EVM" [level=3] [ref=e631]
                - paragraph [ref=e632]: "-16.73 dB"
                - generic [ref=e633]: lower is better
            - region "ACLR_L" [ref=e638]:
              - generic [ref=e639]:
                - heading "ACLR_L" [level=3] [ref=e641]
                - paragraph [ref=e642]: "-29.79 dBc"
                - generic [ref=e643]: lower is better
            - region "ACLR_R" [ref=e648]:
              - generic [ref=e649]:
                - heading "ACLR_R" [level=3] [ref=e651]
                - paragraph [ref=e652]: "-26.35 dBc"
                - generic [ref=e653]: lower is better
            - region "ACLR_AVG" [ref=e658]:
              - generic [ref=e659]:
                - heading "ACLR_AVG" [level=3] [ref=e661]
                - paragraph [ref=e662]: "-28.07 dBc"
                - generic [ref=e663]: lower is better
          - region "Deployment" [ref=e667]:
            - heading "Deployment" [level=2] [ref=e668]
            - paragraph [ref=e669]: "A fixed-point-v1 package: quantised weights, six golden vectors with the state after every sample, a C99 reference verified bit for bit against the software reference, and a report whose numbers say how they were obtained (theoretical, measured execution time, synthesis estimate, measured power). One model, one precision scheme; the rules are pending a maintainer's approval."
            - alert [ref=e670]:
              - generic [ref=e674]: No fixed-point specification for gru. fixed-point-v1 covers gru (one layer, executed as gru_stream); other models have no deployment export yet.
          - generic [ref=e675]:
            - generic [ref=e677]:
              - heading "Reference" [level=2] [ref=e678]
              - paragraph [ref=e679]:
                - code [ref=e680]: measured_pa_output
                - text: — measured PA output of the test split
            - generic [ref=e682]:
              - heading "Models" [level=2] [ref=e683]
              - paragraph [ref=e684]:
                - strong [ref=e685]: pa
                - text: "gru {\"hidden_size\":23,\"num_layers\":1} · 1911 params · offline_segmented"
          - alert [ref=e686]:
            - generic [ref=e690]:
              - strong [ref=e691]: Limitations
              - list [ref=e692]:
                - listitem [ref=e693]: MOCK DATA for UI development; not a computed result
                - listitem [ref=e694]: 3-epoch smoke recipe
          - region "Charts" [ref=e695]:
            - heading "Charts" [level=2] [ref=e696]
            - paragraph [ref=e697]: Drawn from plots-v1 data the worker computed from the evaluated arrays (the spectral profile's Welch estimator). Hiding a trace in the legend changes nothing in the evaluation.
            - paragraph [ref=e698]: No derived plot data for this result (it was evaluated before plots-v1). Re-evaluate the run to produce it.
```

# Test source

```ts
  79  |       await installFakeApi(page, { language })
  80  |       for (const path of ['/', '/datasets', '/experiments', '/experiments/new', '/results', '/settings']) {
  81  |         await page.goto(path)
  82  |         await expect(page.getByRole('main')).toBeVisible()
  83  |         const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  84  |         expect(overflow, `${path} (${language ?? 'en'}) overflows horizontally at ${testInfo.project.name}`).toBeLessThanOrEqual(0)
  85  |       }
  86  |     }
  87  |   })
  88  | 
  89  |   test('unauthenticated session shows the bootstrap explanation, not a blank page', async ({ page }) => {
  90  |     await page.route('**/api/v1/session', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ authenticated: false, csrf_token: null, version: 'x' }) }))
  91  |     await page.goto('/')
  92  |     await expect(page.getByRole('heading', { name: 'Session required' })).toBeVisible()
  93  |     await expect(page.getByLabel('Bootstrap token')).toBeFocused()
  94  |   })
  95  | })
  96  | 
  97  | test.describe('J3 — share and reproduce (mock API)', () => {
  98  |   test('export a share package from a result, then import a package and open the imported run', async ({ page }) => {
  99  |     await installFakeApi(page)
  100 |     await page.goto('/results/run-pa-0001')
  101 |     const panel = page.getByRole('region', { name: 'Export and report' })
  102 |     await expect(panel.getByRole('link', { name: 'Report (HTML)' })).toHaveAttribute('href', '/api/v1/results/run-pa-0001/report?format=html')
  103 |     await expect(panel.getByRole('link', { name: 'Report (Markdown)' })).toHaveAttribute('href', '/api/v1/results/run-pa-0001/report?format=md')
  104 |     await panel.getByRole('button', { name: 'Export share package' }).click()
  105 |     const ready = page.getByTestId('export-ready')
  106 |     await expect(ready).toContainText('Package ready: run-pa-0001-share-20260906.zip')
  107 |     await expect(ready).toContainText('worker logs are not included')
  108 |     await expect(ready.getByRole('link', { name: 'Download package' })).toHaveAttribute('href', '/api/v1/exports/run-pa-0001-share-20260906')
  109 | 
  110 |     await page.goto('/experiments')
  111 |     await page.getByTestId('import-package').setInputFiles({ name: 'run-pa-0001-share.zip', mimeType: 'application/zip', buffer: Buffer.from('zip') })
  112 |     const report = page.getByTestId('import-report')
  113 |     await expect(report).toContainText('Imported run-imported-0001 into this workspace. Dataset capture: missing.')
  114 |     await expect(report).toContainText('dataset capture (raw sha256 …)')
  115 |     await expect(page.getByRole('table', { name: 'Experiments' })).toContainText('imported share package')
  116 |     await report.getByRole('link', { name: 'Open the imported run' }).click()
  117 |     await expect(page).toHaveURL(/\/runs\/run-imported-0001$/)
  118 |     await expect(page.getByRole('heading', { level: 1, name: 'imported share package' })).toBeVisible()
  119 |   })
  120 | })
  121 | 
  122 | test.describe('J2 — my own data (mock API)', () => {
  123 |   test('import CSV with odd headers → doctor → accept estimates → new version → experiment uses it', async ({ page }) => {
  124 |     const state = await installFakeApi(page)
  125 |     await page.goto('/datasets')
  126 |     await page.getByRole('button', { name: 'Advanced import' }).click()
  127 |     const dialog = page.getByRole('dialog')
  128 |     await dialog.getByText('capture.csv').click()
  129 |     await expect(dialog.getByRole('region', { name: 'Column mapping' })).toBeVisible()
  130 |     await expect(dialog.getByLabel('I_out')).toHaveText('rx_i')
  131 |     await expect(dialog.getByLabel('Dataset id')).toHaveValue('capture')
  132 |     await dialog.getByLabel('Sample rate (Hz)').fill('800e6')
  133 |     await dialog.getByLabel('Signal bandwidth (Hz)').fill('200e6')
  134 |     await dialog.getByLabel('Sub-channels').fill('10')
  135 |     await dialog.getByLabel('PSD segment length (nperseg)').fill('2560')
  136 |     await dialog.getByRole('button', { name: 'Import', exact: true }).click()
  137 | 
  138 |     await expect(page).toHaveURL(/\/datasets\/capture$/)
  139 |     await page.getByRole('tab', { name: /Dataset Doctor/ }).click()
  140 |     await page.getByRole('button', { name: 'Run again' }).click()
  141 |     await expect(page.getByRole('article', { name: 'Time misalignment' })).toBeVisible()
  142 | 
  143 |     await page.getByRole('button', { name: 'Preprocess…' }).click()
  144 |     const pre = page.getByRole('dialog')
  145 |     await pre.getByRole('button', { name: 'Use doctor estimates' }).click()
  146 |     await expect(pre.getByLabel('Delay correction (samples)')).toHaveValue('6')
  147 |     await pre.getByRole('button', { name: 'Preview' }).click()
  148 |     await expect(pre.getByRole('article', { name: 'Aligned' })).toBeVisible()
  149 |     await pre.getByLabel('New version name').fill('aligned-v1')
  150 |     await pre.getByRole('button', { name: 'Create version' }).click()
  151 |     await expect(page.getByText('Version aligned-v1 created.')).toBeVisible()
  152 |     await page.getByRole('tab', { name: 'Data versions' }).click()
  153 |     await expect(page.locator('[data-version="aligned-v1"]')).toBeVisible()
  154 | 
  155 |     // the version is a first-class choice when configuring an experiment and travels in the config
  156 |     await page.goto('/experiments/new')
  157 |     await expect(page.getByText('Configuration is valid')).toBeVisible()
  158 |     await page.getByLabel('Data version').click()
  159 |     await page.getByRole('option', { name: 'aligned-v1' }).click()
  160 |     await expect(page.getByText('Configuration is valid')).toBeVisible()
  161 |     await page.getByRole('button', { name: 'Continue' }).click()
  162 |     await page.getByRole('button', { name: 'Continue' }).click()
  163 |     await page.getByRole('button', { name: 'Start run' }).click()
  164 |     await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
  165 |     const config = state.submitted[0]?.['config'] as { dataset: { id: string; preprocessing_version?: string } }
  166 |     expect(config.dataset).toEqual({ id: 'capture', preprocessing_version: 'aligned-v1' })
  167 |   })
  168 | })
  169 | 
  170 | test.describe('component gallery', () => {
  171 |   test('renders real-size charts quickly and matches the visual baseline', async ({ page }, testInfo) => {
  172 |     await installFakeApi(page)
  173 |     await page.goto('/gallery')
  174 |     const probe = page.getByTestId('perf-probe')
  175 |     await expect(probe).toBeVisible({ timeout: 20_000 })
  176 |     const spectrumMs = Number(await probe.getAttribute('data-spectrum-ms'))
  177 |     const iqMs = Number(await probe.getAttribute('data-iq-ms'))
  178 |     test.info().annotations.push({ type: 'perf', description: `spectrum ${spectrumMs} ms, iq ${iqMs} ms` })
> 179 |     expect(spectrumMs + iqMs).toBeLessThan(2000)
      |                               ^ Error: expect(received).toBeLessThan(expected)
  180 |     await expect(page.getByTestId('spectrum-plot').locator('svg.main-svg').first()).toBeVisible()
  181 |     await page.getByRole('button', { name: /Enlarge chart: Power spectral density/ }).click()
  182 |     await expect(page.getByRole('dialog')).toBeVisible()
  183 |     await page.keyboard.press('Escape')
  184 |     await expect(page.getByRole('dialog')).toBeHidden()
  185 |     await page.getByRole('region', { name: 'StatusChip' }).scrollIntoViewIfNeeded()
  186 |     if (testInfo.project.name.startsWith('chromium')) {
  187 |       await expect(page.getByRole('region', { name: 'StatusChip' })).toHaveScreenshot('status-chips.png')
  188 |       await expect(page.getByRole('region', { name: 'MetricCard' })).toHaveScreenshot('metric-cards.png')
  189 |     }
  190 |   })
  191 | })
  192 | 
```