"""Serve a read-only local view: python -m ...search.viewer RUN_DIRECTORY."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import TraceReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()
    reader = TraceReader(args.run_directory)
    page = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            try:
                if route.path == "/":
                    payload, content_type = page, "text/html; charset=utf-8"
                elif route.path == "/api/overview":
                    payload = json.dumps(reader.overview()).encode()
                    content_type = "application/json"
                elif route.path == "/api/frames":
                    values = parse_qs(route.query)
                    data = reader.frames(
                        int(values.get("worker", ["0"])[0]),
                        int(values.get("slot", ["0"])[0]),
                        int(values["tick"][0]) if "tick" in values else None,
                        live_only="tick" not in values,
                    )
                    payload, content_type = json.dumps(data).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, OSError, KeyError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, *_args):
            pass

    print(f"RNA viewer: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
