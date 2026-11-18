# The live monitor: a web page that plays the replay file as if it were a log that a machine is writing right now, lets LogBERT score every
# 5-minute window as soon as it is complete, and raises an alert when LogBERT flags windows. It looks like a monitoring dashboard (live log,
# charts, alert list), because that is what a log-anomaly detector would be used for.
#
#   cd NLP-pipeline
#   make live                       open http://127.0.0.1:8765   (it starts in a few seconds; press Start on the page)
#   python3 demo/live.py --speed 300 --start
#
# It needs only what the pipeline already has (bundle/, processor/engine.py, ../NLP/src, torch, numpy): no Gradio, no cluster, no internet.
# The page itself is live.html; what happens to the lines is in live_core.py. This file is the small web server around them:
#   GET  /                the page
#   GET  /api/feed        what is new since the page last asked (lines, windows, alerts, counters); the page asks about three times a second
#   POST /api/control     {"action": "start" | "pause" | "resume" | "restart", "speed": lines per second}

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from live_core import DEFAULT_SPEED, Monitor

PAGE = Path(__file__).resolve().parent / "live.html"

# How often the program looks at the clock to read the next lines (seconds).
PACE = 0.04

# Uploads are kept in memory for the current replay, with a bounded request size.
MAX_LOG_BYTES = 20 * 1024 * 1024


def number(query, name):
    try:
        return int(query.get(name, ["0"])[0])
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number") from error


def make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        # No line on the terminal for every request: the page asks three times a second.
        def log_message(self, *args):
            pass

        def reply(self, status, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)

            if url.path in ("/", "/index.html"):
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/feed":
                query = parse_qs(url.query)

                try:
                    generation = number(query, "generation") if "generation" in query else None
                    answer = monitor.feed(number(query, "lines"), number(query, "windows"), number(query, "alerts"), generation)
                except ValueError as error:
                    self.reply(400, {"error": str(error)})
                else:
                    self.reply(200, answer)
            else:
                self.reply(404, {"error": "no such address"})

        def do_POST(self):
            path = urlparse(self.path).path
            kind = self.headers.get("Content-Type", "").split(";")[0].strip()
            if path == "/api/upload":
                if kind != "application/octet-stream":
                    self.reply(415, {"error": "Upload a log file as application/octet-stream."})
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.reply(400, {"error": "Invalid upload length."})
                    return
                if size <= 0 or size > MAX_LOG_BYTES:
                    self.reply(413 if size > MAX_LOG_BYTES else 400, {"error": "Choose a non-empty BGL log file up to 20 MB."})
                    return
                try:
                    data = self.rfile.read(size)
                    if len(data) != size:
                        raise ValueError("The upload was incomplete. Try again.")
                    monitor.load_log(data.decode("utf-8-sig"), unquote(self.headers.get("X-Log-Name", "uploaded.log")))
                except (UnicodeError, ValueError) as error:
                    self.reply(400, {"error": str(error)})
                else:
                    self.reply(200, {"dataset": monitor.dataset, "lines": len(monitor.source), "windows": monitor.total_windows})
                return
            if path == "/api/source":
                if kind != "application/json":
                    self.reply(415, {"error": "Send application/json."})
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 1000:
                        raise ValueError("Invalid source request length.")
                    body = json.loads(self.rfile.read(size))
                    if body != {"source": "bundled"}:
                        raise ValueError("The source must be bundled; use Load log file for your own file.")
                    monitor.load_demo()
                except (ValueError, UnicodeError) as error:
                    self.reply(400, {"error": str(error)})
                else:
                    self.reply(200, {"dataset": monitor.dataset, "lines": len(monitor.source), "windows": monitor.total_windows})
                return
            if urlparse(self.path).path != "/api/control":
                self.reply(404, {"error": "no such address"})
                return

            # Only the page itself may send this: a page of another site cannot send JSON without asking the browser's permission first.
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.reply(415, {"error": "send application/json"})
                return

            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(min(size, 1000)) or b"{}")
                state = monitor.control(body.get("action"), body.get("speed"))
            except (ValueError, TypeError, AttributeError) as error:
                self.reply(400, {"error": str(error)})
            else:
                self.reply(200, {"state": state, "speed": monitor.speed})

    return Handler


# The web server (port 0: the system picks a free one, which the tests use).
def make_server(monitor, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), make_handler(monitor))
    server.daemon_threads = True
    return server


# The clock of the replay: every PACE seconds, read the lines that the speed allows (Monitor.tick).
def pace(monitor, stop):
    last = time.monotonic()

    while not stop.is_set():
        time.sleep(PACE)
        now = time.monotonic()
        monitor.tick(now - last)
        last = now


def main(argv=None):
    parser = argparse.ArgumentParser(description="The LogBERT live monitor")
    parser.add_argument("--bundle", help="the folder with drain.pkl, model.pt, settings.json and bgl-replay.log (default: NLP-pipeline/bundle)")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from another computer (there is no login: anyone on the network can look)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="log lines per second (default %(default)s; the page can change it)")
    parser.add_argument("--start", action="store_true", help="start the replay at once instead of waiting for the Start button")
    parser.add_argument("--file", dest="log_file", help="play another BGL-format log file with the saved model")
    args = parser.parse_args(argv)

    print("Loading LogBERT and the replay file...")
    try:
        monitor = Monitor(args.bundle, speed=args.speed, log_file=args.log_file)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Cannot load the log: {error}", file=sys.stderr)
        return 1

    try:
        server = make_server(monitor, args.host, args.port)
    except OSError as error:
        print(f"Cannot open port {args.port} ({error.strerror}). Another program uses it: choose another one with  --port 9000", file=sys.stderr)
        return 1

    if args.start:
        monitor.control("start")

    stop = threading.Event()
    threading.Thread(target=pace, args=(monitor, stop), daemon=True).start()
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"\nOpen  http://{shown}:{args.port}  in your browser.  Stop with Ctrl+C.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop.set()
        server.server_close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
