async (page) => {
  const results = [];
  const check = (pass, label, detail) => { results.push({ label, pass, detail }); if (!pass) throw new Error(JSON.stringify(results)); };
  const near = (a, b) => Math.abs(a - b) < Math.max(1, Math.abs(b)) * 1e-6;
  const state = p => p.evaluate(el => ({ x: [...el._fullLayout.xaxis.range], y: [...el._fullLayout.yaxis.range], mode: el._fullLayout.dragmode }));
  const settled = async () => { await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))); };
  const changed = async (p, before) => {
    await p.evaluate((el, old) => new Promise((resolve, reject) => { const start = performance.now(); const poll = () => { if (el._fullLayout.xaxis.range[0] !== old.x[0] || el._fullLayout.yaxis.range[0] !== old.y[0]) resolve(); else if (performance.now() - start > 5000) reject(new Error('no change')); else requestAnimationFrame(poll); }; poll(); }), before);
    await settled(); return state(p);
  };
  const fit = async p => { await p.getByRole('button', { name: 'Autoscale', exact: true }).click(); await settled(); };
  const exercise = async (p, label) => {
    await p.scrollIntoViewIfNeeded(); await fit(p);
    await p.getByRole('button', { name: 'Pan', exact: true }).click();
    let before = await state(p), r = await p.locator('.nsewdrag').boundingBox();
    await page.mouse.move(r.x + r.width * .6, r.y + r.height * .4); await page.mouse.down();
    await page.mouse.move(r.x + r.width * .45, r.y + r.height * .55, { steps: 8 }); await page.mouse.up();
    let after = await changed(p, before);
    check(after.x[0] > before.x[0] && near(after.x[1] - after.x[0], before.x[1] - before.x[0]), label + ' mouse drag pan');
    check(await p.evaluate(el => document.activeElement === el), label + ' pointer focuses keyboard controls');
    await p.getByRole('button', { name: 'Zoom', exact: true }).click();
    before = await state(p); r = await p.locator('.nsewdrag').boundingBox();
    await page.mouse.move(r.x + r.width * .2, r.y + r.height * .2); await page.mouse.down();
    await page.mouse.move(r.x + r.width * .7, r.y + r.height * .7, { steps: 8 }); await page.mouse.up();
    after = await changed(p, before);
    check(after.x[1] - after.x[0] < before.x[1] - before.x[0] && after.y[1] - after.y[0] < before.y[1] - before.y[0], label + ' box zoom');
    before = after; r = await p.locator('.nsewdrag').boundingBox(); await page.mouse.move(r.x + r.width / 2, r.y + r.height / 2); await page.mouse.wheel(0, -40);
    after = await changed(p, before); check(after.x[1] - after.x[0] < before.x[1] - before.x[0], label + ' unmodified wheel in zoom tool');
    before = after; await p.getByRole('button', { name: 'Zoom out', exact: true }).click(); after = await changed(p, before);
    check(after.x[1] - after.x[0] > before.x[1] - before.x[0], label + ' zoom-out button');
    before = after; await p.getByRole('button', { name: 'Zoom in', exact: true }).click(); after = await changed(p, before);
    check(after.x[1] - after.x[0] < before.x[1] - before.x[0], label + ' zoom-in button');
    await p.focus(); await p.press('p'); await settled(); before = await state(p);
    await p.press('ArrowRight'); after = await changed(p, before); check(after.x[0] > before.x[0] && after.mode === 'pan', label + ' keyboard mode and arrow');
    await p.press('0'); await settled(); before = await state(p); await p.press('+'); after = await changed(p, before);
    check(after.x[1] - after.x[0] < before.x[1] - before.x[0], label + ' keyboard zoom');
    r = await p.locator('.nsewdrag').boundingBox(); await page.mouse.dblclick(r.x + r.width / 2, r.y + r.height / 2); await settled();
    check(await p.evaluate(el => el._fullLayout.xaxis.autorange && el._fullLayout.yaxis.autorange), label + ' double-click fit');
    before = await state(p);
    const webkit = await p.locator('.nsewdrag').evaluate(el => {
      const rect = el.getBoundingClientRect();
      const fire = (type, scale) => { const event = new Event(type, { bubbles: true, cancelable: true }); Object.assign(event, { scale, clientX: rect.x + rect.width / 2, clientY: rect.y + rect.height / 2 }); el.dispatchEvent(event); return event.defaultPrevented; };
      return [fire('gesturestart', 1), fire('gesturechange', 1.25), fire('gesturechange', 1.5), fire('gestureend', 1.5)];
    });
    after = await changed(p, before);
    check(webkit.every(Boolean) && near(after.x[1] - after.x[0], (before.x[1] - before.x[0]) / 1.5), label + ' WebKit gesture-scale path');
    const bypass = await p.locator('.nsewdrag').evaluate(el => {
      const rect = el.getBoundingClientRect();
      return [{ clientX: rect.x + 10, clientY: rect.y + 10, altKey: true }, { clientX: rect.x - 20, clientY: rect.y - 20, ctrlKey: true }].map(options => { const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 30, ...options }); el.dispatchEvent(event); return event.defaultPrevented; });
    });
    check(bypass.every(value => !value), label + ' Alt / outside-axes scroll bypass');
  };
  const inline = page.getByRole('figure', { name: 'Frequency domain', exact: true });
  await exercise(inline, 'inline');
  await page.getByRole('button', { name: 'Enlarge chart: Frequency domain', exact: true }).click();
  const large = page.getByRole('dialog').getByRole('figure'); await large.locator('.nsewdrag').waitFor();
  await exercise(large, 'enlarged');
  const beforeResize = await state(large);
  for (const size of [{ width: 1920, height: 1080 }, { width: 1366, height: 768 }]) {
    await page.setViewportSize(size);
    await large.evaluate(el => new Promise((resolve, reject) => { const start = performance.now(); const poll = () => { if (Math.abs(el._fullLayout.width - el.clientWidth) <= 1 && Math.abs(el._fullLayout.height - el.clientHeight) <= 1) resolve(); else if (performance.now() - start > 5000) reject(new Error('resize did not settle')); else requestAnimationFrame(poll); }; poll(); }));
    const afterResize = await state(large);
    check(afterResize.x.every((v, i) => near(v, beforeResize.x[i])) && afterResize.y.every((v, i) => near(v, beforeResize.y[i])), 'resize keeps camera ' + size.width, afterResize);
  }
  await page.getByRole('button', { name: 'Close enlarged chart', exact: true }).click();
  return results;
}
