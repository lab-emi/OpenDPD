async (page) => {
  const base = page.url().split('/').slice(0,3).join('/');
  const runs = await (await page.request.get(base+'/api/v1/runs')).json();
  const run = runs[0].run_id;
  const routes = ['/', '/datasets', '/datasets/dpa-200mhz', '/experiments', '/experiments/new?task=train_pa', '/runs/'+run, '/results', '/results/'+run, '/settings', '/about'];
  const sizes = [[320,700],[390,844],[640,480],[960,600],[1366,768],[1920,1080],[3840,2160]];
  const results=[];
  for(const route of routes){
    await page.setViewportSize({width:1366,height:768});
    await page.goto(base+route);
    await page.getByRole('heading',{level:1}).first().waitFor();
    await page.locator('main .MuiCircularProgress-root').last().waitFor({state:'hidden'});
    if(route.startsWith('/runs/')) await page.getByTestId('live-dashboard').getByTestId('spectrum-plot').locator('.nsewdrag').waitFor();
    if(route.startsWith('/results/')) await page.getByTestId('spectrum-plot').locator('.nsewdrag').waitFor();
    if(route==='/datasets/dpa-200mhz') await page.getByTestId('spectrum-plot').getByRole('figure').locator('.nsewdrag').waitFor();
    for(const [width,height] of sizes){
      await page.setViewportSize({width,height});
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      await page.waitForFunction(()=>[...document.querySelectorAll('[role="figure"]')].filter(el=>el.clientWidth>0).every(el=>el._fullLayout&&Math.abs(el.clientWidth-el._fullLayout.width)<2),null,{timeout:20000});
      const data=await page.evaluate(()=>({
        viewport:[innerWidth,innerHeight],dpr:devicePixelRatio,bodyWidth:document.documentElement.scrollWidth,
        main:document.querySelector('main').getBoundingClientRect().toJSON(),
        logo:[...document.querySelectorAll('img[alt="OpenDPD Studio"]')].map(el=>({loaded:el.complete&&el.naturalWidth>0,rect:el.getBoundingClientRect().toJSON()})),
        overflows:[...document.querySelectorAll('main *')].filter(el=>{const r=el.getBoundingClientRect();return el.checkVisibility()&&r.width>0&&r.right>innerWidth+2&&!el.closest('.MuiTableContainer-root,[role="figure"],.MuiTabs-root,pre,[aria-hidden="true"]');}).slice(0,8).map(el=>({tag:el.tagName,class:el.className,text:el.textContent.slice(0,100),right:el.getBoundingClientRect().right})),
        charts:[...document.querySelectorAll('[role="figure"]')].map(el=>({width:el.clientWidth,plotWidth:el._fullLayout?.width,types:el._fullData?.map(t=>t.type)}))
      }));
      results.push({route,width,height,...data});
      if((route==='/'||route==='/about'||route==='/datasets'||route==='/datasets/dpa-200mhz'||route==='/experiments'||route.startsWith('/runs/')||route.startsWith('/results/'))&&[390,1366,3840].includes(width)) await page.screenshot({path:'output/playwright/brand-'+(route==='/'?'home':route.slice(1).replaceAll('/','-'))+'-'+width+'.png',scale:'css'});
    }
  }
  return results;
}
