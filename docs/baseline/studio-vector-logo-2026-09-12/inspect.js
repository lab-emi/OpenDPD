async (page) => {
  const base=page.url().split('/').slice(0,3).join('/');
  const results=[];
  for(const [route,width] of [['/about',1366],['/about',390],['/',1366],['/',390]]) {
    await page.setViewportSize({width,height:900});
    await page.goto(base+route); await page.getByRole('heading',{level:1}).first().waitFor();
    await page.waitForFunction(()=>[...document.querySelectorAll('img[alt="OpenDPD Studio"]')].every(el=>el.complete&&el.naturalWidth>0));
    await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
    results.push(await page.evaluate(()=>({path:location.pathname,width:innerWidth,bodyWidth:document.documentElement.scrollWidth,dpr:devicePixelRatio,logos:[...document.querySelectorAll('img[alt="OpenDPD Studio"]')].map(el=>({src:el.src.slice(0,130),dimensions:[el.naturalWidth,el.naturalHeight],bounds:el.getBoundingClientRect().toJSON(),parentBackground:getComputedStyle(el.parentElement).backgroundColor}))})));
    await page.screenshot({path:'output/playwright/vector-'+(route==='/about'?'about':'home')+'-'+width+'.png',scale:'device'});
  }
  await page.goto(base+'/about');
  await page.waitForFunction(()=>[...document.querySelectorAll('img[alt="OpenDPD Studio"]')].length===2&&[...document.querySelectorAll('img[alt="OpenDPD Studio"]')].every(el=>el.complete&&el.naturalWidth>0));
  const alpha=await page.evaluate(async()=>{
    const img=document.querySelector('main img[alt="OpenDPD Studio"]');
    const svg=await (await fetch(img.src)).text();
    const doc=new DOMParser().parseFromString(svg,'image/svg+xml');
    const canvas=document.createElement('canvas');canvas.width=2400;canvas.height=672;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(img,0,0,2400,672);
    const at=(x,y)=>[...ctx.getImageData(x*4,y*4,1,1).data];
    const pixels=ctx.getImageData(0,0,2400,672).data;
    let transparent=0,opaque=0,antialiased=0;
    for(let i=3;i<pixels.length;i+=4){if(pixels[i]===0)transparent++;else if(pixels[i]===255)opaque++;else antialiased++;}
    return {embeddedRaster:doc.querySelectorAll('image,foreignObject').length,fontText:doc.querySelectorAll('text').length,externalReferences:doc.querySelectorAll('[href],[style]').length,paths:doc.querySelectorAll('path').length,corners:[at(0,0),at(599,0),at(0,167),at(599,167)],terminalHoles:[at(142,53),at(149,87),at(142,121)],wordCounter:at(213,62),transparent,opaque,antialiased};
  });
  if(alpha.embeddedRaster||alpha.fontText||alpha.corners.some(p=>p[3]!==0)||alpha.terminalHoles.some(p=>p[3]!==0)||alpha.wordCounter[3]!==0||!alpha.antialiased)throw new Error(JSON.stringify(alpha));
  return {screens:results,alpha};
}
