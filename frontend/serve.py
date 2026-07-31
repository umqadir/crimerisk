"""
Static dev server for the CrimeRisk frozen-snapshot map.

Serves frontend/public/ with HTTP Range support so MapLibre's pmtiles protocol
can range-request tiles directly out of the single .pmtiles archive. pmtiles.js
REQUIRES 206 Partial Content responses; the stdlib SimpleHTTPRequestHandler does
not implement Range, so we add a minimal single-range handler here. No build
step, no node — just `uv run python frontend/serve.py [port] [dir]`.

The optional second argument is the directory to serve; it defaults to the public
dir next to this script, but a frozen snapshot staged elsewhere can be served for
review by pointing it at that directory.

Run:  uv run python frontend/serve.py 8777 [dir]
"""

import os
import re
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
PUBLIC = (
    Path(sys.argv[2]).resolve()
    if len(sys.argv) > 2
    else Path(__file__).resolve().parent / "public"
)

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class Handler(SimpleHTTPRequestHandler):
    # HTTP/1.1 is required for Range / 206 to be honored by clients.
    protocol_version = "HTTP/1.1"

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".pmtiles": "application/octet-stream",
        ".json": "application/json",
        ".js": "application/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()

        m = _RANGE_RE.fullmatch(rng.strip())
        if not m:
            return super().do_GET()

        size = os.path.getsize(path)
        start_s, end_s = m.group(1), m.group(2)
        if start_s == "" and end_s == "":
            return super().do_GET()
        if start_s == "":  # suffix range: last N bytes
            length = int(end_s)
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        ctype = self.guess_type(path)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    if not PUBLIC.exists():
        raise SystemExit(f"public dir not found: {PUBLIC} (run the build first)")
    handler = partial(Handler, directory=str(PUBLIC))
    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"Serving {PUBLIC} at http://127.0.0.1:{PORT}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
