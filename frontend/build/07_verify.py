"""07 - Serve the staged site, drive it in a real browser and record evidence.

Captures the required screenshots at 1440x900 and 390x844, measures the first
data paint (first frame after a choropleth tile loaded; the pre-2025.2 metric,
PMTiles header arrival, is recorded beside it) and the bytes transferred to
reach the national view, and writes the numbers next to the screenshots.

States covered beyond the plain map: a block group below its denominator floor
(hatch, card text, the explicit "Show tract estimate" action), a shard that
fails to load (load error, distinct from no data), and the mobile bottom sheet
in its collapsed, controls and detail snaps with the selected polygon kept in
the unobscured part of the map.

Run:  uv run python frontend/build/07_verify.py [port]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crschema import DIST, REPO, TILES_ROOT  # noqa: E402

from playwright.sync_api import sync_playwright  # noqa: E402

SITE = DIST / "site"
SHOTS = TILES_ROOT / "screenshots"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
BASE = f"http://127.0.0.1:{PORT}/"

DESKTOP = {"width": 1440, "height": 900}
MOBILE = {"width": 390, "height": 844}

CHICAGO_LOOP = "11.40/41.8800/-87.6300"
ILLINOIS = "6.60/40.0000/-89.2000"
# A zero-resident Loop block group whose exposure and per-resident indexes are
# both suppressed in the 2025 surface (status 1 on every published measure).
SUPPRESSED_BG = "170313201012"
SUPPRESSED_VIEW = "14.00/41.8831/-87.6172"

# Is the shown selection inside the part of the map the sheet or card leaves free?
SELECTION_VISIBLE_JS = """() => {
  const map = window.__crimeriskMap;
  const sel = document.querySelector('#sidebar').getBoundingClientRect();
  const wrap = document.querySelector('.map-wrap').getBoundingClientRect();
  const u = new URL(location.href);
  const s = (u.searchParams.get('s') || '').split(':');
  if (s.length !== 2) return {selected: false};
  const src = {county: 'c', tract: 't', bg: 'b'}[s[0]];
  const layer = {county: 'counties', tract: 'tracts', bg: 'blockgroups'}[s[0]];
  const feats = map.querySourceFeatures(src, {sourceLayer: layer, filter: ['==', ['get', 'geoid'], s[1]]});
  if (!feats.length) return {selected: true, found: false};
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const f of feats) {
    const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const poly of polys) for (const pt of poly[0]) {
      const p = map.project(pt);
      x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
    }
  }
  const mobile = window.matchMedia('(max-width: 700px)').matches;
  let right = wrap.width, bottom = wrap.height;
  if (mobile) bottom = sel.top - wrap.top;
  for (const id of ['card', 'compare']) {
    const el = document.getElementById(id);
    if (!mobile && el.classList.contains('on')) right = Math.min(right, el.getBoundingClientRect().left - wrap.left);
  }
  return {selected: true, found: true, bbox_px: [x0, y0, x1, y1].map(Math.round),
          free_px: [0, 0, Math.round(right), Math.round(bottom)],
          inside: x0 >= 0 && y0 >= 0 && x1 <= right && y1 <= bottom};
}"""


class Recorder:
    def __init__(self, page):
        self.total = 0
        self.by_kind = {}
        page.on("response", self._on)

    def _on(self, resp):
        try:
            length = resp.headers.get("content-length")
            n = int(length) if length else len(resp.body())
        except Exception:
            return
        self.total += n
        url = resp.url
        kind = ("tiles" if ".pmtiles" in url else
                "shards" if "/shards/" in url else
                "bench" if "/bench/" in url else
                "basemap" if "openfreemap" in url else
                "app")
        self.by_kind[kind] = self.by_kind.get(kind, 0) + n


def wait_map(page, ms=30000):
    page.wait_for_function(
        "() => window.__crimeriskFirstPaint !== undefined", timeout=ms
    )
    page.wait_for_timeout(2200)


def shot(page, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / name))
    print(f"  {name}")


def paint_numbers(page) -> dict:
    return {
        "first_data_paint_ms": page.evaluate("window.__crimeriskFirstPaint"),
        "first_metadata_ms": page.evaluate("window.__crimeriskFirstMetadata"),
        "script_start_ms": page.evaluate("window.__crimeriskScriptStart"),
    }


def card_state(page) -> dict:
    return {
        "status": page.get_attribute("#card", "data-status"),
        "text": page.inner_text("#card"),
    }


def attempt(results, key, fn):
    """Steps that depend on the public geocoder are recorded, not fatal."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        results.setdefault("step_errors", {})[key] = str(exc).splitlines()[0][:300]
        print(f"  step {key} failed: {results['step_errors'][key]}")


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, str(REPO / "frontend" / "serve.py"), str(PORT), str(SITE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.2)
    results = {}
    console_errors = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()

            # --- cold national load, desktop: paint time and bytes ---------
            ctx = browser.new_context(viewport=DESKTOP, device_scale_factor=2)
            page = ctx.new_page()
            page.on("console", lambda m: console_errors.append(("desktop", m.text)) if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(("desktop", str(e))))
            rec = Recorder(page)
            t0 = time.time()
            page.goto(BASE, wait_until="domcontentloaded")
            wait_map(page)
            results["desktop_national"] = {
                **paint_numbers(page),
                "wall_to_paint_ms": int((time.time() - t0) * 1000),
                "bytes_total": rec.total,
                "bytes_by_kind": dict(rec.by_kind),
            }
            shot(page, "01_national_desktop_1440x900.png")

            # --- state view -------------------------------------------------
            page.goto(BASE + f"?m=exposure&x=ov&v={ILLINOIS}", wait_until="domcontentloaded")
            wait_map(page)
            shot(page, "02_state_illinois_desktop_1440x900.png")

            # --- Chicago neighborhood, Overall ------------------------------
            rec2_start = rec.total
            page.goto(BASE + f"?m=exposure&x=ov&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
            wait_map(page)
            results["chicago_overall_bytes"] = rec.total - rec2_start
            shot(page, "03_chicago_overall_desktop_1440x900.png")

            # --- special-use annotation + hover chip ------------------------
            page.check("#special-toggle")
            page.wait_for_timeout(1500)
            page.mouse.move(900, 500)
            page.wait_for_timeout(1200)
            results["hover_text"] = page.inner_text("#hover")
            shot(page, "09_special_use_and_hover_desktop_1440x900.png")
            page.uncheck("#special-toggle")

            # --- Chicago neighborhood, Murder (tract-only) ------------------
            page.goto(BASE + f"?m=exposure&x=mur&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
            wait_map(page)
            shot(page, "04_chicago_murder_desktop_1440x900.png")

            # --- suppressed block group: hatch, card, explicit tract action --
            page.goto(
                BASE + f"?m=exposure&x=ov&s=bg:{SUPPRESSED_BG}&v={SUPPRESSED_VIEW}",
                wait_until="domcontentloaded",
            )
            wait_map(page)
            page.wait_for_selector("#card.on", timeout=15000)
            page.wait_for_timeout(1200)
            results["suppressed_card"] = card_state(page)
            shot(page, "18_suppressed_card_desktop_1440x900.png")
            page.click("#card .actions button")
            page.wait_for_function(
                "() => /Tract estimate/i.test(document.getElementById('card').innerText)", timeout=15000
            )
            page.wait_for_timeout(1200)
            results["tract_estimate_card"] = card_state(page)
            shot(page, "19_show_tract_estimate_desktop_1440x900.png")

            # --- a selection under the card is moved into the free area ------
            page.goto(BASE + "?m=exposure&x=ov&v=12.00/41.9000/-87.7400", wait_until="domcontentloaded")
            wait_map(page)
            page.mouse.click(1200, 250)
            page.wait_for_selector("#card.on", timeout=15000)
            page.wait_for_timeout(1500)
            results["desktop_selection_under_card"] = page.evaluate(SELECTION_VISIBLE_JS)
            shot(page, "10_selection_fitted_desktop_1440x900.png")

            # --- search-selected card ---------------------------------------
            def search_card():
                page.goto(BASE + f"?m=exposure&x=ov&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
                wait_map(page)
                page.fill("#q", "233 S Wacker Dr, Chicago, IL")
                page.click("#search-form button[type=submit]")
                page.wait_for_selector("#card.on", timeout=25000)
                page.wait_for_timeout(2500)
                results["card_text"] = page.inner_text("#card")
                shot(page, "05_search_card_desktop_1440x900.png")

            attempt(results, "search_card", search_card)

            # --- compare ----------------------------------------------------
            def compare():
                page.click("#card .links button")
                page.wait_for_selector("#compare.on", timeout=8000)
                page.fill("#q", "1060 W Addison St, Chicago, IL")
                page.click("#search-form button[type=submit]")
                page.wait_for_selector("#compare table", timeout=30000)
                page.wait_for_timeout(1500)
                results["compare_text"] = page.inner_text("#compare")
                results["compare_url"] = page.url
                shot(page, "06_compare_desktop_1440x900.png")

            attempt(results, "compare", compare)

            # --- methods ------------------------------------------------------
            page.goto(BASE + "methods.html", wait_until="load")
            page.wait_for_timeout(700)
            shot(page, "07_methods_desktop_1440x900.png")
            page.goto(BASE + "download.html", wait_until="load")
            page.wait_for_timeout(700)
            shot(page, "08_download_desktop_1440x900.png")
            ctx.close()

            # --- load error: shard requests fail -----------------------------
            ectx = browser.new_context(viewport=DESKTOP, device_scale_factor=2)
            ectx.route(re.compile(r".*/data/shards/bg/(?!index).*\.json"), lambda r: r.fulfill(status=503, body="down"))
            epage = ectx.new_page()
            epage.goto(
                BASE + f"?m=exposure&x=ov&s=bg:{SUPPRESSED_BG}&v={SUPPRESSED_VIEW}",
                wait_until="domcontentloaded",
            )
            wait_map(epage)
            epage.wait_for_selector("#card.on", timeout=15000)
            epage.wait_for_timeout(800)
            results["load_error_card"] = card_state(epage)
            shot(epage, "20_load_error_card_desktop_1440x900.png")
            ectx.unroute(re.compile(r".*/data/shards/bg/(?!index).*\.json"))
            epage.click("#card .actions button")
            epage.wait_for_function(
                "() => document.getElementById('card').getAttribute('data-status') !== 'error'", timeout=15000
            )
            results["load_error_retry_status"] = epage.get_attribute("#card", "data-status")
            ectx.close()

            # --- mobile -------------------------------------------------------
            mctx = browser.new_context(
                viewport=MOBILE, device_scale_factor=3, is_mobile=True, has_touch=True
            )
            mpage = mctx.new_page()
            mpage.on("console", lambda m: console_errors.append(("mobile", m.text)) if m.type == "error" else None)
            mpage.on("pageerror", lambda e: console_errors.append(("mobile", str(e))))
            mrec = Recorder(mpage)
            mpage.goto(BASE, wait_until="domcontentloaded")
            wait_map(mpage)
            results["mobile_national"] = {
                **paint_numbers(mpage),
                "bytes_total": mrec.total,
                "bytes_by_kind": dict(mrec.by_kind),
            }
            shot(mpage, "11_national_mobile_390x844.png")

            mpage.goto(BASE + f"?m=exposure&x=ov&v={ILLINOIS}", wait_until="domcontentloaded")
            wait_map(mpage)
            shot(mpage, "12_state_illinois_mobile_390x844.png")

            mpage.goto(BASE + f"?m=exposure&x=ov&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
            wait_map(mpage)
            results["mobile_sheet_collapsed_px"] = mpage.evaluate(
                "Math.round(document.getElementById('sidebar').getBoundingClientRect().height)"
            )
            shot(mpage, "13_chicago_overall_mobile_390x844.png")

            mpage.click('#sheet-tabs button[data-snap="controls"]')
            mpage.wait_for_timeout(600)
            results["mobile_sheet_controls_px"] = mpage.evaluate(
                "Math.round(document.getElementById('sidebar').getBoundingClientRect().height)"
            )
            shot(mpage, "21_controls_sheet_mobile_390x844.png")
            mpage.click('#sheet-tabs button[data-snap="collapsed"]')
            mpage.wait_for_timeout(400)

            # A tap low on the map: the detail sheet opens over that spot, and the
            # selected polygon has to end up above the sheet.
            mpage.touchscreen.tap(195, 640)
            mpage.wait_for_selector("#card.on", timeout=15000)
            mpage.wait_for_timeout(1800)
            results["mobile_sheet_detail_px"] = mpage.evaluate(
                "Math.round(document.getElementById('sidebar').getBoundingClientRect().height)"
            )
            results["mobile_selection_above_sheet"] = mpage.evaluate(SELECTION_VISIBLE_JS)
            results["mobile_selected_url"] = mpage.url
            shot(mpage, "15_selected_card_mobile_390x844.png")

            mpage.goto(BASE + f"?m=exposure&x=mur&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
            wait_map(mpage)
            shot(mpage, "14_chicago_murder_mobile_390x844.png")

            mpage.goto(
                BASE + f"?m=resident&x=ov&s=bg:{SUPPRESSED_BG}&v={SUPPRESSED_VIEW}",
                wait_until="domcontentloaded",
            )
            wait_map(mpage)
            mpage.wait_for_selector("#card.on", timeout=15000)
            mpage.wait_for_timeout(1500)
            results["mobile_suppressed_card"] = card_state(mpage)
            results["mobile_suppressed_above_sheet"] = mpage.evaluate(SELECTION_VISIBLE_JS)
            shot(mpage, "22_suppressed_card_mobile_390x844.png")

            if results.get("compare_url"):
                from urllib.parse import urlparse, parse_qs, urlencode

                q = parse_qs(urlparse(results["compare_url"]).query)
                flat = {k: v[0] for k, v in q.items()}
                mpage.goto(BASE + "?" + urlencode(flat), wait_until="domcontentloaded")
                wait_map(mpage)
                mpage.wait_for_timeout(2500)
                shot(mpage, "16_compare_mobile_390x844.png")

            mpage.goto(BASE + "methods.html", wait_until="load")
            mpage.wait_for_timeout(600)
            shot(mpage, "17_methods_mobile_390x844.png")
            mctx.close()
            browser.close()
    finally:
        server.terminate()

    results["console_errors"] = console_errors
    (SHOTS / "measurements.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:6000])


if __name__ == "__main__":
    main()
