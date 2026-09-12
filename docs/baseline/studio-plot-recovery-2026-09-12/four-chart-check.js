async (page) => {
  const results = [];
  for (const id of ['spectrum-plot', 'iq-preview', 'constellation-plot', 'dataset-am-plot']) {
    const figure = page.getByTestId(id).getByRole('figure');
    const read = () => figure.evaluate(el => ({ x: [...el._fullLayout.xaxis.range], y: [...el._fullLayout.yaxis.range], autoX: !!el._fullLayout.xaxis.autorange, autoY: !!el._fullLayout.yaxis.autorange, count: el._fullData.reduce((sum, tr) => sum + tr.x.length, 0), types: el._fullData.map(tr => tr.type) }));
    await figure.locator('.nsewdrag').waitFor({ state: 'visible' });
    const before = await read();
    const rect = await figure.locator('.nsewdrag').boundingBox();
    await page.mouse.move(rect.x + rect.width / 2, rect.y + rect.height / 2);
    await page.mouse.wheel(8, 0);
    await page.waitForFunction(id => { const el = document.querySelector(`[data-testid="${id}"] [role="figure"]`); return !el._fullLayout.xaxis.autorange && !el._fullLayout.yaxis.autorange; }, id);
    const smallPan = await read();
    await page.waitForTimeout(1100);
    const retained = await read();
    if (retained.autoX || retained.autoY || retained.x.some((v, i) => v !== smallPan.x[i])) throw new Error(`${id}: valid pan was reset`);
    await page.mouse.wheel(rect.width * 4, rect.height * 2);
    await page.waitForFunction(({id, x}) => { const el = document.querySelector(`[data-testid="${id}"] [role="figure"]`); return el._fullLayout.xaxis.range[0] > x; }, { id, x: Math.max(...smallPan.x) });
    const outside = await read();
    if (id === 'spectrum-plot') await page.screenshot({ path: 'output/playwright/plot-recovery-before.png', scale: 'css' });
    await page.waitForFunction(id => { const el = document.querySelector(`[data-testid="${id}"] [role="figure"]`); return el._fullLayout.xaxis.autorange && el._fullLayout.yaxis.autorange; }, id, { timeout: 8000 });
    const fitted = await read();
    if (fitted.count !== before.count) throw new Error(`${id}: displayed sample count changed`);
    if (id === 'spectrum-plot') await page.screenshot({ path: 'output/playwright/plot-recovery-after.png', scale: 'css' });
    results.push({ id, before, smallPan, retained, outside, fitted });
  }
  return results;
}
