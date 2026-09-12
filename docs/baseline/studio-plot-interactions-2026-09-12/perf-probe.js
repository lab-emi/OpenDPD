async (page) => {
  const reports = [];
  const chinese = await page.evaluate(() => document.documentElement.lang.startsWith('zh'));
  const enlargeLabel = chinese ? '放大图表' : 'Enlarge chart', closeLabel = chinese ? '关闭大屏图表' : 'Close enlarged chart';
  for (const id of ['spectrum-plot', 'constellation-plot']) {
    const title = await page.getByTestId(id).getByRole('figure').getAttribute('aria-label');
    for (const enlarged of [false, true]) {
      if (enlarged) await page.getByRole('button', { name: enlargeLabel + ': ' + title, exact: true }).click();
      const plot = enlarged ? page.getByRole('dialog').getByRole('figure', { name: title, exact: true }) : page.getByRole('figure', { name: title, exact: true });
      await plot.locator('.nsewdrag').waitFor(); await plot.scrollIntoViewIfNeeded();
      await plot.getByRole('button', { name: 'Pan', exact: true }).click();
      const report = await plot.evaluate(async el => {
        const bundle = el.data.some(trace => trace.type === 'scattergl') ? 'plotly-strict.min-' : 'plotly-basic.min-';
        const url = performance.getEntriesByType('resource').find(entry => entry.name.includes(bundle)).name;
        const api = (await import(url)).default, original = api.relayout, durations = [];
        let active = 0, maxConcurrent = 0, calls = 0, events = 0;
        api.relayout = function (element, ...args) {
          if (element !== el) return original.call(this, element, ...args);
          active++; calls++; maxConcurrent = Math.max(active, maxConcurrent);
          const started = performance.now();
          return Promise.resolve(original.call(this, element, ...args)).finally(() => { active--; durations.push(performance.now() - started); });
        };
        const start = performance.now(), x = [...el._fullLayout.xaxis.range], y = [...el._fullLayout.yaxis.range];
        const spanX = x[1] - x[0], spanY = y[1] - y[0];
        let lastInput = start;
        try {
          for (let frame = 0; frame < 60; frame++) {
            await new Promise(resolve => requestAnimationFrame(resolve));
            const area = el.querySelector('.nsewdrag'), rect = area.getBoundingClientRect();
            for (let i = 0; i < 4; i++) {
              area.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: rect.x + rect.width / 2, clientY: rect.y + rect.height / 2, deltaX: .25, deltaY: .1 }));
              events++; x[0] += spanX * .25 / rect.width; y[0] -= spanY * .1 / rect.height;
            }
            lastInput = performance.now();
          }
          await new Promise((resolve, reject) => {
            const poll = () => {
              if (!active && Math.abs(el._fullLayout.xaxis.range[0] - x[0]) < 1e-6 && Math.abs(el._fullLayout.yaxis.range[0] - y[0]) < 1e-6) resolve();
              else if (performance.now() - lastInput > 5000) reject(new Error('input backlog did not drain'));
              else requestAnimationFrame(poll);
            }; poll();
          });
          durations.sort((a, b) => a - b);
          return { events, redraws: calls, maxConcurrent, elapsedMs: performance.now() - start, finalInputDrainMs: performance.now() - lastInput, medianDrawMs: durations[Math.floor(durations.length / 2)], p95DrawMs: durations[Math.floor(durations.length * .95)], plottedPoints: el.data.reduce((total, trace) => total + trace.x.length, 0), pass: events === 240 && calls < events && maxConcurrent === 1 };
        } finally { api.relayout = original; }
      });
      reports.push({ title, enlarged, ...report });
      if (!report.pass) throw new Error(JSON.stringify(reports));
      if (enlarged) await page.getByRole('button', { name: closeLabel, exact: true }).click();
    }
  }
  return reports;
}
