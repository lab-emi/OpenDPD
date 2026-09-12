import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'

const running = JSON.parse(readFileSync(new URL('../mocks/run_running.json', import.meta.url), 'utf8')) as { data: Record<string, unknown> }

test('web polling survives a dropped connection and reload, restores plots, and Stop has no shell input', async ({ page }) => {
  let offline = false
  let revision = 1
  let status = 'running'
  const cursors: number[] = []
  const cancellations: string[] = []
  const id = 'web-reconnect-run'
  const record = () => ({ ...running.data, run_id: id, name: 'Remote CUDA experiment', device: 'cuda', status })
  await page.addInitScript(() => sessionStorage.setItem('opendpd-web-session:https://api.opendpd.com', 'browser-test-capability'))
  await page.route('https://api.opendpd.com/api/v1/**', async (route) => {
    const request = route.request(), url = new URL(request.url()), path = url.pathname.slice('/api/v1'.length)
    if (offline) return route.abort('internetdisconnected')
    let body: unknown = {}
    if (path === '/session') body = { authenticated: true, mode: 'web', version: '2.2.0.dev0', expires_at: new Date(Date.now() + 3600000).toISOString() }
    else if (path === '/settings') body = { language: 'en' }
    else if (path === '/system/capabilities') body = { version: '2.2.0.dev0', workspace: 'Temporary workspace', note: '', devices: [{ device: 'cpu', detected: true }, { device: 'cuda', detected: true, name: 'GPU', count: 1 }], custom_dataset_imports: false }
    else if (path === '/runs') body = [record()]
    else if (path === '/runs/count') body = { count: 1 }
    else if (path === `/runs/${id}`) body = record()
    else if (path.endsWith('/events/list')) {
      const after = Number(url.searchParams.get('after') ?? 0)
      cursors.push(after)
      body = { last_seq: revision, terminal: status === 'cancelled', events: after < revision ? [{ seq: revision, run_id: id, type: 'metric', ts: new Date().toISOString(), payload: { epoch: revision, split: 'val', values: { NMSE: -30 - revision } } }] : [] }
    } else if (path.endsWith('/live')) body = {
      policy: { min_batches: 25, min_seconds: 2, overhead_target: .05 }, geometry: null,
      preview: { revision, updated_at: new Date().toISOString(), source: 'validation_probe', samples: 3,
        metrics: { NMSE: -30 - revision }, units: { NMSE: 'dB' }, plots: {
          spectrum: { frequency: [-1e6, 0, 1e6], traces: [{ name: 'Output', psd_db: [-50, -10, -50] }] },
          time: { start: 0, traces: [{ name: 'Output', i: [0, 1, 0], q: [1, 0, -1] }] },
        } },
    }
    else if (path.endsWith('/history')) body = []
    else if (path.endsWith('/lineage')) body = { parents: [], children: [] }
    else if (path.endsWith('/logs')) body = { lines: url.searchParams.get('offset') === '0' ? ['CUDA worker is running'] : [], next_offset: 23, eof: true, size: 23 }
    else if (path.endsWith('/cancel')) {
      expect(request.headers()['content-type']).toBe('application/json')
      expect(request.headers()['authorization']).toBe('Bearer browser-test-capability')
      cancellations.push(path)
      status = 'cancelled'
      body = record()
    }
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body), headers: { 'Access-Control-Allow-Origin': 'http://127.0.0.1:4174' } })
  })
  await page.goto(`/#/runs/${id}`)
  await expect(page.getByRole('heading', { name: 'Remote CUDA experiment', exact: true })).toBeVisible()
  const plots = page.getByTestId('live-dashboard').locator('.js-plotly-plot')
  await expect(plots).toHaveCount(3)
  await expect(page.getByText('-31.00 dB', { exact: true })).toBeVisible()
  offline = true
  await page.getByRole('button', { name: 'Resync experiment' }).click()
  await expect(page.getByText(/disconnected/i).first()).toBeVisible({ timeout: 15_000 })
  // Already rendered experiment/plot data survives failed background reads.
  await expect(page.getByRole('heading', { name: 'Remote CUDA experiment', exact: true })).toBeVisible()
  await expect(plots).toHaveCount(3)
  revision = 2
  offline = false
  await page.getByRole('button', { name: 'Resync experiment' }).click()
  await expect(page.getByText('-32.00 dB', { exact: true })).toBeVisible()
  expect(cursors).toContain(1)
  await page.reload()
  await expect(page.getByText('-32.00 dB', { exact: true })).toBeVisible()
  await expect(plots).toHaveCount(3)
  await page.getByRole('button', { name: /Terminal.*Running/ }).click()
  const terminal = page.getByTestId('experiment-terminal')
  await expect(terminal.getByRole('log')).toContainText('CUDA worker is running')
  await expect(terminal.getByRole('textbox')).toHaveCount(1)
  await expect(terminal.getByRole('textbox')).toHaveAccessibleName('Filter lines')
  await terminal.getByRole('button', { name: 'Stop experiment' }).click()
  await expect.poll(() => cancellations).toEqual([`/runs/${id}/cancel`])
  await expect(terminal.getByRole('button', { name: 'Stop experiment' })).toHaveCount(0)
})
