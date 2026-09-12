# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: a11y.spec.ts >> keyboard-only journey >> complete the dataset guide and start PA training without a pointer
- Location: e2e/a11y.spec.ts:91:3

# Error details

```
Error: no focusable element named /^Continue$/ within 80 tabs
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
      - paragraph [ref=e60]: Experiments
      - paragraph [ref=e62]: opendpd workspace
      - button [ref=e63] [cursor=pointer]
      - button [ref=e67] [cursor=pointer]
      - generic [ref=e71]: Nothing running
      - button "Language" [ref=e73] [cursor=pointer]:
        - generic [ref=e75]: English
  - main [ref=e79]:
    - form [ref=e80]:
      - generic [ref=e81]:
        - heading "PA Model Training" [level=1] [ref=e82]
        - button "Advanced JSON" [ref=e83] [cursor=pointer]
      - navigation "Choose a task" [ref=e84]:
        - link "01 PA Model Training" [ref=e85] [cursor=pointer]:
          - /url: /experiments/new?task=train_pa&dataset=dpa-200mhz&version=raw-v1
          - generic [ref=e86]: "01"
          - paragraph [ref=e92]: PA Model Training
        - link "02 PA Model Testing" [ref=e93] [cursor=pointer]:
          - /url: /experiments/new?task=evaluate_pa&dataset=dpa-200mhz&version=raw-v1
          - generic [ref=e94]: "02"
          - paragraph [ref=e100]: PA Model Testing
        - link "03 DPD Model Training" [ref=e101] [cursor=pointer]:
          - /url: /experiments/new?task=train_dpd&dataset=dpa-200mhz&version=raw-v1
          - generic [ref=e102]: "03"
          - paragraph [ref=e108]: DPD Model Training
        - link "04 DPD Model Testing" [ref=e109] [cursor=pointer]:
          - /url: /experiments/new?task=run_dpd&dataset=dpa-200mhz&version=raw-v1
          - generic [ref=e110]: "04"
          - paragraph [ref=e116]: DPD Model Testing
      - tablist "Experiment setup steps" [ref=e119]:
        - tab [ref=e120] [cursor=pointer]:
          - generic [ref=e121]:
            - img "Complete" [ref=e122]
            - text: Data & model
        - tab "2 Model & training" [selected] [ref=e124] [cursor=pointer]:
          - generic [ref=e125]:
            - generic [ref=e126]: "2"
            - text: Model & training
        - tab "3 Review & run" [disabled]:
          - generic:
            - generic: "3"
            - text: Review & run
      - alert [ref=e127]:
        - generic [ref=e131]: "Quick trial: a few epochs to check the workflow. Results are not a benchmark."
      - tabpanel "2 Model & training" [ref=e132]:
        - generic [ref=e134]:
          - generic [ref=e136]:
            - generic [ref=e137]: Device
            - generic [ref=e138]:
              - combobox "Device" [ref=e139] [cursor=pointer]: cpu
              - textbox [aria-hidden]: cpu
              - group [aria-hidden]:
                - generic: Device
            - paragraph [ref=e140]
          - generic [ref=e142]:
            - generic [ref=e143]: Seed
            - generic [ref=e144]:
              - spinbutton "Seed" [ref=e145]
              - group [aria-hidden]:
                - generic: Seed
            - paragraph [ref=e146]
          - generic [ref=e148]:
            - generic: Name (optional)
            - generic [ref=e149]:
              - textbox "Name (optional)" [ref=e150]
              - group [aria-hidden]:
                - generic: Name (optional)
            - paragraph [ref=e151]
        - generic [ref=e152]:
          - heading "Training hyperparameters" [level=2] [ref=e153]
          - generic [ref=e154]:
            - generic [ref=e156]:
              - generic [ref=e157]: Epochs
              - generic [ref=e158]:
                - spinbutton "Epochs" [ref=e159]
                - group [aria-hidden]:
                  - generic: Epochs
              - paragraph [ref=e160]
            - generic [ref=e162]:
              - generic [ref=e163]: Batch size
              - generic [ref=e164]:
                - spinbutton "Batch size" [ref=e165]
                - group [aria-hidden]:
                  - generic: Batch size
              - paragraph [ref=e166]
            - generic [ref=e168]:
              - generic [ref=e169]: Learning rate
              - generic [ref=e170]:
                - spinbutton "Learning rate" [ref=e171]
                - group [aria-hidden]:
                  - generic: Learning rate
              - paragraph [ref=e172]
            - generic [ref=e174]:
              - generic [ref=e175]: Frame length
              - generic [ref=e176]:
                - spinbutton "Frame length" [ref=e177]
                - group [aria-hidden]:
                  - generic: Frame length
              - paragraph [ref=e178]
            - generic [ref=e180]:
              - generic [ref=e181]: Frame stride
              - generic [ref=e182]:
                - spinbutton "Frame stride" [ref=e183]
                - group [aria-hidden]:
                  - generic: Frame stride
              - paragraph [ref=e184]
            - generic [ref=e186]:
              - generic [ref=e187]: Optimizer
              - generic [ref=e188]:
                - combobox "Optimizer" [ref=e189] [cursor=pointer]: adamw
                - textbox [aria-hidden]: adamw
                - group [aria-hidden]:
                  - generic: Optimizer
            - generic [ref=e191]:
              - generic [ref=e192]: Loss function
              - generic [ref=e193]:
                - combobox "Loss function" [ref=e194] [cursor=pointer]: L2
                - textbox [aria-hidden]: l2
                - group [aria-hidden]:
                  - generic: Loss function
        - generic [ref=e195]:
          - heading [level=3] [ref=e196]:
            - button [ref=e197] [cursor=pointer]:
              - paragraph [ref=e199]: Advanced settings
          - generic: Metric profile
          - generic: CPU threads (optional)
      - alert [ref=e203]:
        - generic [ref=e207]:
          - generic [ref=e208]: Warnings
          - list [ref=e209]:
            - listitem [ref=e210]:
              - code [ref=e211]: training.epochs
              - text: ": smoke length"
      - generic [ref=e212]:
        - button [ref=e213] [cursor=pointer]
        - button [ref=e217] [cursor=pointer]
        - link "Cancel" [active] [ref=e221] [cursor=pointer]:
          - /url: /experiments
        - status [ref=e222]: Configuration is valid
```

# Test source

```ts
  1   | import { AxeBuilder } from '@axe-core/playwright'
  2   | import { expect, test, type Page } from '@playwright/test'
  3   | import { installFakeApi } from './mock-api'
  4   | 
  5   | /**
  6   |  * S13 accessibility and offline checks against the mocked API.
  7   |  *  - axe-core (WCAG 2.1 A/AA rules) on every main page: no serious or critical violation;
  8   |  *  - a keyboard-only journey from the home page to a started run;
  9   |  *  - the main journey needs no host other than the loopback one (no fonts, CDN or telemetry).
  10  |  */
  11  | 
  12  | const PAGES = ['/', '/datasets', '/datasets?guide=start', '/experiments', '/experiments/new', '/experiments/new?task=evaluate_pa', '/experiments/new?task=train_dpd', '/experiments/new?task=run_dpd', '/results', '/results/run-pa-0001', '/settings', '/gallery']
  13  | 
  14  | async function scan(page: Page, path: string) {
  15  |   // audit the settled page: queries answered and MUI's colour transitions (~300 ms) finished
  16  |   await page.waitForLoadState('networkidle')
  17  |   await page.waitForTimeout(600)
  18  |   const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
  19  |   const blocking = results.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')
  20  |   const summary = results.violations.map((v) => `${v.impact}: ${v.id} (${v.nodes.length} nodes) — ${v.help}`)
  21  |   test.info().annotations.push({ type: `axe ${path}`, description: summary.length ? summary.join('; ') : 'no violations' })
  22  |   expect(blocking.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`), `${path} has blocking accessibility violations`).toEqual([])
  23  | }
  24  | 
  25  | test.describe('accessibility (axe-core)', () => {
  26  |   test('main pages have no serious or critical WCAG 2.1 AA violations', async ({ page }) => {
  27  |     await installFakeApi(page)
  28  |     for (const path of PAGES) {
  29  |       await page.goto(path)
  30  |       await expect(page.getByRole(path.includes('guide=') ? 'dialog' : 'main')).toBeVisible()
  31  |       if (path === '/gallery') await expect(page.getByTestId('spectrum-plot').locator('svg.main-svg').first()).toBeVisible({ timeout: 20_000 })
  32  |       await scan(page, path)
  33  |     }
  34  |   })
  35  | 
  36  |   test('run detail tabs are accessible while a run is live', async ({ page }) => {
  37  |     await installFakeApi(page)
  38  |     await page.goto('/datasets')
  39  |     await page.getByRole('button', { name: 'Built-in datasets' }).click()
  40  |     await page.getByRole('button', { name: 'Add & inspect DPA_200MHz' }).click()
  41  |     await page.goto('/experiments/new')
  42  |     await expect(page.getByText('Configuration is valid')).toBeVisible()
  43  |     await page.getByRole('button', { name: 'Continue' }).click()
  44  |     await page.getByRole('button', { name: 'Continue' }).click()
  45  |     await page.getByRole('button', { name: 'Start run' }).click()
  46  |     await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
  47  |     await scan(page, '/runs/run-e2e-0001')
  48  |     for (const tab of ['Logs', 'Artifacts', 'Configuration']) {
  49  |       await page.getByRole('tab', { name: tab }).click()
  50  |       await scan(page, `/runs/run-e2e-0001#${tab}`)
  51  |     }
  52  |   })
  53  | })
  54  | 
  55  | /** Press Tab until the focused element has the given accessible name (bounded so a trap fails the test). */
  56  | async function tabTo(page: Page, name: RegExp, maxTabs = 80): Promise<void> {
  57  |   for (let i = 0; i < maxTabs; i++) {
  58  |     await page.keyboard.press('Tab')
  59  |     const label = await page.evaluate(() => {
  60  |       const el = document.activeElement as HTMLElement | null
  61  |       if (!el) return ''
  62  |       return el.getAttribute('aria-label') ?? el.textContent?.trim() ?? ''
  63  |     })
  64  |     if (name.test(label)) return
  65  |   }
> 66  |   throw new Error(`no focusable element named ${name} within ${maxTabs} tabs`)
      |         ^ Error: no focusable element named /^Continue$/ within 80 tabs
  67  | }
  68  | 
  69  | test.describe('keyboard-only journey', () => {
  70  |   test('page reset returns focus and global reset keeps focus in the new guide', async ({ page }) => {
  71  |     await installFakeApi(page)
  72  |     await page.goto('/experiments')
  73  |     await tabTo(page, /^Reset page$/)
  74  |     await page.keyboard.press('Enter')
  75  |     await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused()
  76  |     await page.keyboard.press('Enter')
  77  |     await expect(page.getByRole('button', { name: 'Reset page', exact: true })).toBeFocused()
  78  |     await tabTo(page, /^Reset Studio$/)
  79  |     await page.keyboard.press('Enter')
  80  |     await tabTo(page, /^Reset$/)
  81  |     await page.keyboard.press('Enter')
  82  |     await expect(page.getByRole('heading', { name: 'Start the guided setup again?' })).toHaveCount(0)
  83  |     const guide = page.getByRole('dialog')
  84  |     await expect(guide).toContainText('Create your first dataset')
  85  |     await expect.poll(() => guide.evaluate((element) => element.contains(document.activeElement))).toBe(true)
  86  |     await tabTo(page, /^Skip tutorial$/)
  87  |     await page.keyboard.press('Enter')
  88  |     await expect(guide).toHaveCount(0)
  89  |   })
  90  | 
  91  |   test('complete the dataset guide and start PA training without a pointer', async ({ page }) => {
  92  |     const state = await installFakeApi(page)
  93  |     await page.goto('/')
  94  |     await expect(page.getByRole('link', { name: 'Get Started' })).toBeVisible()
  95  |     await tabTo(page, /^Get Started$/)
  96  |     await page.keyboard.press('Enter')
  97  |     await expect(page.getByRole('dialog')).toBeVisible()
  98  |     await tabTo(page, /^Try a built-in dataset$/)
  99  |     await page.keyboard.press('Enter')
  100 |     await expect(page.getByRole('button', { name: 'Add & inspect DPA_200MHz' })).toBeVisible()
  101 |     await tabTo(page, /^Add & inspect DPA_200MHz$/)
  102 |     await page.keyboard.press('Enter')
  103 |     await expect(page.getByRole('button', { name: 'Inspect my dataset' })).toBeVisible()
  104 |     await tabTo(page, /^Inspect my dataset$/)
  105 |     await page.keyboard.press('Enter')
  106 |     await expect(page.getByRole('link', { name: 'Configure experiment' })).toBeEnabled()
  107 |     await tabTo(page, /^Configure experiment$/)
  108 |     await page.keyboard.press('Enter')
  109 |     await expect(page.getByRole('heading', { level: 1, name: 'PA Model Training' })).toBeVisible()
  110 |     await expect(page.getByText('Configuration is valid')).toBeVisible()
  111 |     await tabTo(page, /^Continue$/)
  112 |     await page.keyboard.press('Enter')
  113 |     await tabTo(page, /^Continue$/)
  114 |     await page.keyboard.press('Enter')
  115 |     await tabTo(page, /^Start run$/)
  116 |     await page.keyboard.press('Enter')
  117 |     await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
  118 |     expect(state.submitted).toHaveLength(1)
  119 |     // tabs follow the ARIA tabs pattern: Tab reaches the selected tab, arrows move between tabs
  120 |     await tabTo(page, /^Overview$/)
  121 |     await page.keyboard.press('ArrowRight')
  122 |     await page.keyboard.press('Enter')
  123 |     await expect(page.getByRole('tab', { name: 'Logs' })).toHaveAttribute('aria-selected', 'true')
  124 |     await expect(page.getByRole('tabpanel')).toContainText('Training Completed...')
  125 |   })
  126 | })
  127 | 
  128 | test.describe('offline', () => {
  129 |   test('the main journey contacts no host but the loopback one', async ({ page, context }) => {
  130 |     const external: string[] = []
  131 |     await context.route('**/*', (route) => {
  132 |       const url = new URL(route.request().url())
  133 |       if (url.hostname === '127.0.0.1' || url.hostname === 'localhost') return route.continue()
  134 |       external.push(url.href)
  135 |       return route.abort()
  136 |     })
  137 |     await installFakeApi(page)
  138 |     await page.goto('/datasets')
  139 |     await page.getByRole('button', { name: 'Built-in datasets' }).click()
  140 |     await page.getByRole('button', { name: 'Add & inspect DPA_200MHz' }).click()
  141 |     await page.goto('/experiments/new')
  142 |     await page.getByRole('button', { name: 'Continue' }).click()
  143 |     await page.getByRole('button', { name: 'Continue' }).click()
  144 |     await page.getByRole('button', { name: 'Start run' }).click()
  145 |     await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
  146 |     await page.goto('/results/run-pa-0001')
  147 |     await expect(page.getByRole('region', { name: 'Export and report' })).toBeVisible()
  148 |     await page.goto('/gallery')     // charts: the Plotly bundle and every asset come from this origin
  149 |     await expect(page.getByTestId('spectrum-plot').locator('svg.main-svg').first()).toBeVisible({ timeout: 20_000 })
  150 |     expect(external).toEqual([])
  151 |   })
  152 | })
  153 | 
```