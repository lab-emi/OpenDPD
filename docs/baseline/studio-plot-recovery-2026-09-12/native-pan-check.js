async (page) => {
  const figure = page.getByTestId('spectrum-plot').getByRole('figure');
  const rect = await figure.locator('.nsewdrag').boundingBox();
  await page.mouse.move(rect.x + rect.width - 5, rect.y + rect.height / 2);
  await page.mouse.down();
  await page.mouse.move(rect.x + 5, rect.y + rect.height / 2, { steps: 10 });
  await page.waitForTimeout(1100);
  await page.mouse.up();
  await page.waitForFunction(() => !document.querySelector('[data-testid="spectrum-plot"] [role="figure"]')._fullLayout.xaxis.autorange);
  const moved = await figure.evaluate(el => {
    const x = el._fullLayout.xaxis.range, y = el._fullLayout.yaxis.range;
    let visible = 0, total = 0;
    for (const tr of el._fullData) for (let i = 0; i < tr.x.length; i++) { total++; if (tr.x[i] >= x[0] && tr.x[i] <= x[1] && tr.y[i] >= y[0] && tr.y[i] <= y[1]) visible++; }
    return { x: [...x], y: [...y], visible, total };
  });
  if (moved.visible <= 0 || moved.visible >= moved.total) throw new Error('Did not produce a partially empty native-pan view');
  await page.waitForFunction(() => document.querySelector('[data-testid="spectrum-plot"] [role="figure"]')._fullLayout.xaxis.autorange, undefined, { timeout: 8000 });
  return { moved, fitted: await figure.evaluate(el => ({ x: el._fullLayout.xaxis.range, y: el._fullLayout.yaxis.range, autoX: el._fullLayout.xaxis.autorange, autoY: el._fullLayout.yaxis.autorange })) };
}
