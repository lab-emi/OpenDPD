async (page) => {
  const results = [];
  const chinese = await page.evaluate(() => document.documentElement.lang.startsWith('zh'));
  const enlargeLabel = chinese ? '放大图表' : 'Enlarge chart';
  const closeLabel = chinese ? '关闭大屏图表' : 'Close enlarged chart';
  await page.evaluate(() => {
    window.__gestureQaWheel = [];
    document.addEventListener('wheel', event => window.__gestureQaWheel.push({ x: event.clientX, y: event.clientY, dx: event.deltaX, dy: event.deltaY }), { capture: true });
  });
  const check = (pass, label, detail) => { results.push({ label, pass, detail }); if (!pass) throw new Error(JSON.stringify(results)); };
  const near = (a, b) => Math.abs(a - b) < Math.max(1, Math.abs(b)) * 1e-6;
  const rangesEqual = (a, b) => a.x.every((v, i) => near(v, b.x[i])) && a.y.every((v, i) => near(v, b.y[i]));
  const state = (plot) => plot.evaluate(el => ({ x: [...el._fullLayout.xaxis.range], y: [...el._fullLayout.yaxis.range], autoX: el._fullLayout.xaxis.autorange, autoY: el._fullLayout.yaxis.autorange, mode: el._fullLayout.dragmode }));
  const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const changed = async (plot, before) => {
    await plot.evaluate((el, old) => new Promise((resolve, reject) => {
      const started = performance.now();
      const poll = () => {
        const now = el._fullLayout;
        if (now && (now.xaxis.range[0] !== old.x[0] || now.yaxis.range[0] !== old.y[0])) resolve();
        else if (performance.now() - started > 8000) reject(new Error('viewport did not change'));
        else requestAnimationFrame(poll);
      }; poll();
    }), before);
    await settle();
    return state(plot);
  };
  const exercise = async (plot, label) => {
    await plot.scrollIntoViewIfNeeded();
    const before = await state(plot);
    let rect = await plot.locator('.nsewdrag').boundingBox();
    await page.mouse.move(rect.x + rect.width * .3, rect.y + rect.height * .4);
    await page.mouse.wheel(13.75, -6.25);
    const panned = await changed(plot, before);
    check(near(panned.x[1] - panned.x[0], before.x[1] - before.x[0]) && near(panned.y[1] - panned.y[0], before.y[1] - before.y[0]) && panned.x[0] > before.x[0] && panned.y[0] > before.y[0], label + ' diagonal fractional pan', { before, panned });
    rect = await plot.locator('.nsewdrag').boundingBox();
    await page.mouse.move(rect.x + rect.width * .3, rect.y + rect.height * .4);
    await page.keyboard.down('Control');
    await page.mouse.wheel(0, -60);
    await page.keyboard.up('Control');
    const zoomed = await changed(plot, panned);
    const wheel = await page.evaluate(() => window.__gestureQaWheel.at(-1));
    const ax = (wheel.x - rect.x) / rect.width, ay = 1 - (wheel.y - rect.y) / rect.height;
    const anchor = (range, fraction) => range[0] + (range[1] - range[0]) * fraction;
    check(zoomed.x[1] - zoomed.x[0] < panned.x[1] - panned.x[0] && near(anchor(zoomed.x, ax), anchor(panned.x, ax)) && near(anchor(zoomed.y, ay), anchor(panned.y, ay)), label + ' pinch/Ctrl-wheel cursor zoom', { panned, zoomed, wheel });
    return zoomed;
  };
  await page.waitForFunction(() => document.querySelectorAll('.js-plotly-plot .nsewdrag').length === 4);
  const titles = await page.locator('main .js-plotly-plot').evaluateAll(els => els.map(el => el.getAttribute('aria-label')));
  for (const title of titles) {
    const inline = page.getByRole('figure', { name: title, exact: true });
    const inlineView = await exercise(inline, title + ' inline');
    await page.getByRole('button', { name: enlargeLabel + ': ' + title, exact: true }).click();
    const enlarged = page.getByRole('dialog').getByRole('figure', { name: title, exact: true });
    await enlarged.locator('.nsewdrag').waitFor();
    check(rangesEqual(inlineView, await state(enlarged)), title + ' keeps viewport when enlarged', await state(enlarged));
    const largeView = await exercise(enlarged, title + ' enlarged');
    await page.getByRole('button', { name: closeLabel, exact: true }).click();
    await inline.locator('.nsewdrag').waitFor(); await settle();
    check(rangesEqual(largeView, await state(inline)), title + ' keeps viewport on close', await state(inline));
    await inline.focus(); await inline.press('Home');
    await inline.evaluate(el => new Promise(resolve => { const poll = () => el._fullLayout.xaxis.autorange && el._fullLayout.yaxis.autorange ? resolve() : requestAnimationFrame(poll); poll(); }));
    check((await state(inline)).autoX === true && (await state(inline)).autoY === true, title + ' keyboard fit');
  }
  return results;
}
