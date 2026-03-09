import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
fs.mkdirSync(path.join(__dirname, 'thumbnails'), { recursive: true });

const ids = [
  'waermepumpe-wolf',
  'waermepumpe-wolf-mit-bodenkonsole',
  'waermepumpe-wolf-bodenkonsole-abstaende',
];

const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
const page = await browser.newPage();

page.on('console', m => { if (m.type() !== 'log') console.log('browser:', m.text()); });
page.on('pageerror', e => console.error('pageerror:', e.message));

await page.goto('http://localhost:8080/thumb-gen.html');
console.log('Seite geladen, warte auf Rendering…');

await page.waitForSelector('#done', { timeout: 60000 });
console.log('Alle Modelle gerendert, speichere PNGs…');

const data = await page.evaluate(() => window._thumbData);

for (const id of ids) {
  const b64 = data[id].replace(/^data:image\/png;base64,/, '');
  const outPath = path.join(__dirname, 'thumbnails', id + '.png');
  fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
  const size = fs.statSync(outPath).size;
  console.log(`✓ ${id}.png (${Math.round(size/1024)} KB)`);
}

await browser.close();
console.log('Fertig!');
