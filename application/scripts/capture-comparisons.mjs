import {chromium} from '@playwright/test';
import fs from 'node:fs';
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:1536,height:1024},deviceScaleFactor:1});
const items=[
 ['reference-pricing','/home/ram/Downloads/ChatGPT Image Sep 18, 2026, 08_20_50 AM.png',{x:490,y:309,width:1028,height:271}],
 ['implementation-pricing','qa/application-desktop.png',{x:479,y:316,width:1039,height:257}],
 ['reference-models','/home/ram/Downloads/ChatGPT Image Sep 18, 2026, 08_20_34 AM.png',{x:231,y:727,width:422,height:224}],
 ['implementation-models','qa/dashboard-desktop.png',{x:230,y:738,width:408,height:235}],
 ['reference-offer','/home/ram/Downloads/ChatGPT Image Sep 18, 2026, 07_56_56 AM.png',{x:576,y:442,width:520,height:379}],
 ['implementation-offer','qa/offer-desktop.png',{x:580,y:445,width:530,height:424}],
];
for(const [name,path,clip] of items){
 const data=fs.readFileSync(path).toString('base64');
 await page.setContent(`<style>body{margin:0}img{display:block;width:1536px;height:1024px}</style><img src="data:image/png;base64,${data}">`);
 await page.locator('img').evaluate(img=>img.decode());
 await page.screenshot({path:`qa/${name}-detail.png`,clip});
}
await browser.close();
