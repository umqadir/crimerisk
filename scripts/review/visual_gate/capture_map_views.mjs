// Visual-gate capture harness for the CrimeRisk map.
//
// Drives the frozen-snapshot viewer with Playwright and captures a named set of
// views (layer + center + zoom) as PNGs for the supervisor's release visual
// gate. Hardening carried over from the blind-sniff program's lessons: the
// control panel is hidden only at screenshot time, every shot asserts rendered
// features > 0 (one reload retry — a server without Range support once produced
// ten silently blank captures), and a colored-pixel fraction is asserted by the
// Python wrapper that ships alongside this file.
//
// Usage: node capture_map_views.mjs <baseURL> <outDir> <suffix> [viewsJson]
//   baseURL   e.g. http://localhost:8778 (serve.py pointed at a snapshot dir)
//   outDir    directory for PNGs
//   suffix    appended to view names, e.g. "v19"
//   viewsJson optional path to a views file; defaults to views.json next to
//             this script

import { chromium } from "playwright";
import { readFileSync, mkdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const [baseURL, outDir, suffix, viewsPath] = process.argv.slice(2);
if (!baseURL || !outDir || !suffix) {
  console.error("usage: node capture_map_views.mjs <baseURL> <outDir> <suffix> [viewsJson]");
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));
const views = JSON.parse(readFileSync(viewsPath || join(here, "views.json"), "utf8"));
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

async function renderedFeatureCount() {
  return page.evaluate(() => {
    const layers = ["bg-fill", "tract-fill"].filter(l => map.getLayer(l));
    return map.queryRenderedFeatures({ layers }).length;
  });
}

async function settle() {
  await page.waitForFunction(() => typeof map !== "undefined" && map.loaded(), null, { timeout: 60000 });
  await page.waitForTimeout(1200);
}

let failures = 0;
for (const v of views) {
  const name = `${v.name}_${suffix}.png`;
  for (let attempt = 0; attempt < 2; attempt++) {
    await page.goto(baseURL, { waitUntil: "domcontentloaded" });
    await settle();
    await page.evaluate(({ layer, center, zoom }) => {
      if (layer) setLayer(layer);
      map.jumpTo({ center, zoom });
    }, v);
    await page.waitForFunction(() => map.loaded(), null, { timeout: 60000 });
    await page.waitForTimeout(1500);
    const n = await renderedFeatureCount();
    if (n > 0) {
      await page.addStyleTag({ content: ".panel { display: none !important; }" });
      await page.screenshot({ path: join(outDir, name) });
      console.log(`ok ${name} features=${n}`);
      break;
    }
    if (attempt === 1) {
      failures++;
      console.error(`FAIL ${name}: zero rendered features after retry`);
    }
  }
}

await browser.close();
process.exit(failures ? 1 : 0);
