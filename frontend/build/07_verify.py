"""07 - Serve the staged site, drive it in a real browser and record evidence.

Captures the required screenshots at 1440x900 and 390x844, measures the first
meaningful data paint and the bytes transferred to reach the national view, and
writes the numbers next to the screenshots.

Run:  uv run python frontend/build/07_verify.py [port]
"""

from __future__ import annotations

import json
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


def wait_map(page, ms=20000):
    page.wait_for_function(
        "() => window.__crimeriskFirstPaint !== undefined", timeout=ms
    )
    page.wait_for_timeout(2200)


def shot(page, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / name))
    print(f"  {name}")


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, str(REPO / "frontend" / "serve.py"), str(PORT), str(SITE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.2)
    results = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()

            # --- cold national load, desktop: paint time and bytes ---------
            ctx = browser.new_context(viewport=DESKTOP, device_scale_factor=2)
            page = ctx.new_page()
            rec = Recorder(page)
            t0 = time.time()
            page.goto(BASE, wait_until="domcontentloaded")
            wait_map(page)
            paint = page.evaluate("window.__crimeriskFirstPaint")
            results["desktop_national"] = {
                "first_data_paint_ms": paint,
                "script_start_ms": page.evaluate("window.__crimeriskScriptStart"),
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

            # --- search-selected card ---------------------------------------
            page.goto(BASE + f"?m=exposure&x=ov&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
            wait_map(page)
            page.fill("#q", "233 S Wacker Dr, Chicago, IL")
            page.click("#search-form button[type=submit]")
            page.wait_for_selector("#card.on", timeout=25000)
            page.wait_for_timeout(2500)
            results["card_text"] = page.inner_text("#card")
            shot(page, "05_search_card_desktop_1440x900.png")

            # --- compare ----------------------------------------------------
            page.click("#card .links button")
            page.wait_for_selector("#compare.on", timeout=8000)
            page.fill("#q", "1060 W Addison St, Chicago, IL")
            page.click("#search-form button[type=submit]")
            page.wait_for_selector("#compare table", timeout=30000)
            page.wait_for_timeout(1500)
            results["compare_text"] = page.inner_text("#compare")
            results["compare_url"] = page.url
            shot(page, "06_compare_desktop_1440x900.png")

            # --- methods ------------------------------------------------------
            page.goto(BASE + "methods.html", wait_until="load")
            page.wait_for_timeout(700)
            shot(page, "07_methods_desktop_1440x900.png")
            page.goto(BASE + "download.html", wait_until="load")
            page.wait_for_timeout(700)
            shot(page, "08_download_desktop_1440x900.png")
            ctx.close()

            # --- mobile -------------------------------------------------------
            mctx = browser.new_context(
                viewport=MOBILE, device_scale_factor=3, is_mobile=True, has_touch=True
            )
            mpage = mctx.new_page()
            mrec = Recorder(mpage)
            mpage.goto(BASE, wait_until="domcontentloaded")
            wait_map(mpage)
            results["mobile_national"] = {
                "first_data_paint_ms": mpage.evaluate("window.__crimeriskFirstPaint"),
                "bytes_total": mrec.total,
                "bytes_by_kind": dict(mrec.by_kind),
            }
            shot(mpage, "11_national_mobile_390x844.png")

            mpage.goto(BASE + f"?m=exposure&x=ov&v={ILLINOIS}", wait_until="domcontentloaded")
            wait_map(mpage)
            shot(mpage, "12_state_illinois_mobile_390x844.png")

            mpage.goto(BASE + f"?m=exposure&x=ov&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
            wait_map(mpage)
            shot(mpage, "13_chicago_overall_mobile_390x844.png")

            mpage.goto(BASE + f"?m=exposure&x=mur&v={CHICAGO_LOOP}", wait_until="domcontentloaded")
            wait_map(mpage)
            shot(mpage, "14_chicago_murder_mobile_390x844.png")

            if results.get("compare_url"):
                from urllib.parse import urlparse, parse_qs, urlencode

                q = parse_qs(urlparse(results["compare_url"]).query)
                flat = {k: v[0] for k, v in q.items()}
                mpage.goto(BASE + "?" + urlencode(flat), wait_until="domcontentloaded")
                wait_map(mpage)
                mpage.wait_for_timeout(2500)
                shot(mpage, "16_compare_mobile_390x844.png")
                flat.pop("k", None)
                mpage.goto(BASE + "?" + urlencode(flat), wait_until="domcontentloaded")
                wait_map(mpage)
                mpage.wait_for_timeout(2000)
                shot(mpage, "15_selected_card_mobile_390x844.png")

            mpage.goto(BASE + "methods.html", wait_until="load")
            mpage.wait_for_timeout(600)
            shot(mpage, "17_methods_mobile_390x844.png")
            mctx.close()
            browser.close()
    finally:
        server.terminate()

    (SHOTS / "measurements.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:4000])


if __name__ == "__main__":
    main()
