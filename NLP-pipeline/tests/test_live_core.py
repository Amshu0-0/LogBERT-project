# The live monitor (demo/live_core.py, demo/live.py, demo/live.html): the back end on the real bundle, the small web server around it, and the page
# as far as it can be checked without a browser. The replay of all 30,000 lines is run once (about 10 seconds) and shared by the tests.
#
# What must hold: the monitor reaches the very numbers of the pipeline's own run (evidence/local-replay.json: 394 windows, 116 anomalous, 266 flagged),
# because it uses the same Windower and the same LogBERT calculation as the processor; every flagged window belongs to exactly one alert; and the
# page asks for nothing that the server does not give.

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE / "demo"))

import live  # noqa: E402
import live_core  # noqa: E402
from demo_core import Row, window_ranks  # noqa: E402
from live_core import MAX_PER_ANSWER, Monitor  # noqa: E402

BUNDLE = PIPELINE / "bundle"
PAGE = PIPELINE / "demo" / "live.html"

pytestmark = pytest.mark.skipif(not (BUNDLE / "model.pt").is_file(), reason="the bundle is not here: run make export-bundle")


def quiet():
    return {"say": lambda *_: None}


@pytest.fixture(scope="module")
def finished():
    monitor = Monitor(**quiet())
    monitor.run_all()
    return monitor


# ------------------------------------------------------------------------------------------------ the numbers

def test_the_replay_gives_the_numbers_of_the_pipeline_s_own_run(finished):
    saved = json.loads((PIPELINE / "evidence" / "local-replay.json").read_text(encoding="utf-8"))
    logbert = saved["detectors"]["logbert"]
    counts = finished.counts

    assert finished.state == "finished" and finished.position == len(finished.source) == 30000
    assert counts["lines"] == saved["lines"]["stored"] and counts["alert_lines"] == saved["lines"]["with_alert_tag"]
    assert counts["windows"] == saved["windows"] == finished.total_windows == 394
    assert counts["tp"] + counts["fn"] == saved["anomalous_windows"] == 116
    assert (counts["tp"], counts["fp"], counts["tn"], counts["fn"]) == (logbert["tp"], logbert["fp"], logbert["tn"], logbert["fn"])
    assert counts["flagged"] == counts["tp"] + counts["fp"] == 266


def test_every_verdict_is_the_one_the_processor_would_give(finished):
    # An independent cut into windows (the NLP project's own function) and the processor's Detectors.score(), window by window.
    from logad.data.windows import sliding_windows

    ids = np.array([row.template_id for row in finished.rows], dtype=np.int64)
    pairs, _ = sliding_windows(np.array([row.ts for row in finished.rows], dtype=np.int64), 300, 60)

    assert len(pairs) == len(finished.windows)

    for number in range(0, len(pairs), 3):
        lo, hi = pairs[number]
        score = finished.detectors.score(ids[lo:hi])
        window = finished.windows[number]

        assert (window["lines"], window["missed"], window["flagged"]) == (hi - lo, score.missed, score.flag), number


def test_the_ranks_of_a_window_come_from_the_same_calculation_as_its_verdict(finished):
    rng = np.random.default_rng(5)
    keys = finished.detectors.logbert.vocab.keys

    for length in (1, 7, 60, 300, 600):
        ids = rng.choice(keys, size=length)
        worst, ranks = window_ranks(finished.detectors, ids)

        assert worst == finished.detectors.score(ids).missed and len(ranks) == length
        assert all(rank is not None for rank in ranks), "the full-coverage variant hides every key once"

    assert window_ranks(finished.detectors, []) == (0, [])


# ------------------------------------------------------------------------------------------------ the alerts

def test_every_flagged_window_is_in_one_alert_and_the_alerts_do_not_overlap(finished):
    flagged = [w["number"] for w in finished.windows if w["flagged"]]
    covered = []

    for alert in finished.incidents:
        inside = [n for n in flagged if alert["first_window"] <= n <= alert["last_window"]]

        assert len(inside) == alert["windows"] == 1
        assert alert["state"] == "replay_ended" and alert["peak"] > finished.r and alert["limit"] == finished.r
        covered += inside

    assert covered == flagged
    assert [a["id"] for a in finished.incidents] == list(range(1, len(finished.incidents) + 1))
    assert all(a["last_window"] < b["first_window"] for a, b in zip(finished.incidents, finished.incidents[1:], strict=False))
    assert finished.open is None


def test_an_alert_is_real_when_its_windows_hold_lines_with_an_alert_tag(finished):
    # One alert per flagged window: the counts match the saved true and false positives.
    real = [a for a in finished.incidents if a["real"]]

    assert (len(finished.incidents), len(real), len(finished.incidents) - len(real)) == (266, 112, 154)

    for alert in finished.incidents:
        tagged = sum(w["alert_lines"] for w in finished.windows[alert["first_window"] - 1: alert["last_window"]])
        assert alert["real"] == (alert["alert_lines"] > 0) and (tagged > 0) == alert["real"]


def test_an_alert_names_the_message_types_that_the_model_could_not_guess(finished):
    for alert in finished.incidents:
        reasons = alert["reasons"]

        assert 1 <= len(reasons) <= live_core.TOP_REASONS
        assert [r["count"] for r in reasons] == sorted((r["count"] for r in reasons), reverse=True)
        assert all(r["template"] and r["example"] and r["count"] > 0 for r in reasons)


def test_unflagged_windows_do_not_close_alerts_and_new_flags_get_their_own_alert():
    monitor = Monitor.__new__(Monitor)
    monitor.incidents, monitor.incident_seq, monitor.open = [], 0, None
    monitor.g, monitor.r, monitor.known, monitor.log_time = 8, 8, set(), 1000
    monitor.rows = [Row(1000, "KERNDTLB", "new message", -1, "new message")]
    window = SimpleNamespace(start=1000, end=1300)
    monitor._alert({"flagged": 1, "number": 1, "missed": 9, "lines": 1}, window, [0], [-1], [9])
    first = monitor.incidents[0]

    for number in range(2, 102):
        monitor._alert({"flagged": 0, "number": number}, window, [], [], [])

    assert first["state"] == "firing" and monitor.open is first
    monitor._alert({"flagged": 1, "number": 102, "missed": 10, "lines": 1}, window, [0], [-1], [9])
    assert len(monitor.incidents) == 2 and all(a["state"] == "firing" for a in monitor.incidents)
    monitor._end_replay()
    assert monitor.open is None
    assert all(a["state"] == "replay_ended" and a["end_of_log"] and "resolved" not in a for a in monitor.incidents)


# ------------------------------------------------------------------------------------------------ playing, pausing, restarting

def test_nothing_is_read_until_the_replay_is_started_and_pausing_stops_it():
    monitor = Monitor(speed=1000, **quiet())

    assert monitor.state == "ready" and monitor.tick(1.0) == 0 and monitor.position == 0
    assert monitor.control("start") == "running"
    assert monitor.tick(0.1) == 100 and monitor.position == 100
    assert monitor.control("pause") == "paused" and monitor.tick(5.0) == 0 and monitor.position == 100
    assert monitor.control("resume") == "running"
    assert monitor.tick(0.05) == 50 and monitor.position == 150


def test_a_long_pause_of_the_clock_is_made_up_for_by_two_seconds_of_lines_at_most():
    monitor = Monitor(speed=100, **quiet())
    monitor.control("start")

    assert monitor.tick(60.0) == 200


def test_the_speed_can_be_changed_and_is_kept_within_limits():
    monitor = Monitor(**quiet())

    monitor.control(None, 300)
    assert monitor.speed == 300
    monitor.control(None, 0)
    assert monitor.speed == 1
    monitor.control(None, 10 ** 9)
    assert monitor.speed == 5000

    with pytest.raises(ValueError, match="unknown action"):
        monitor.control("explode")


def test_a_restart_starts_from_nothing_and_gives_the_same_windows_again():
    monitor = Monitor(**quiet())
    monitor.control("start")
    monitor.step(3000)
    first = [{k: v for k, v in w.items() if k != "ms"} for w in monitor.windows]
    generation = monitor.generation

    assert monitor.control("restart") == "running"
    assert monitor.generation == generation + 1 and monitor.position == 0 and monitor.windows == [] and monitor.incidents == [] and not monitor.lines
    assert monitor.counts["lines"] == 0

    monitor.step(3000)

    assert [{k: v for k, v in w.items() if k != "ms"} for w in monitor.windows] == first


def test_the_end_of_the_log_flushes_the_last_windows_and_closes_the_open_alert(finished):
    # Completing playback changes every alert to replay_ended without claiming recovery.
    assert finished.counts["windows"] == finished.total_windows
    assert all(a["end_of_log"] is True and a["state"] == "replay_ended" for a in finished.incidents)
    assert finished.feed()["state"] == "finished" and finished.feed()["firing"] is False


# ------------------------------------------------------------------------------------------------ what the page gets

def test_the_feed_gives_only_what_is_new_and_can_be_sent_as_json():
    monitor = Monitor(**quiet())
    monitor.control("start")
    monitor.step(2000)

    first = monitor.feed()
    json.dumps(first)

    assert first["reset"] is False and first["state"] == "running" and first["total_lines"] == 30000 and first["total_windows"] == 394
    assert first["counts"]["lines"] == 2000 and len(first["lines"]) == MAX_PER_ANSWER and first["lines"][-1]["seq"] == 2000
    assert [line["seq"] for line in first["lines"]] == list(range(2000 - MAX_PER_ANSWER + 1, 2001))
    assert len(first["windows"]) == monitor.counts["windows"] > 5 and first["alerts"]
    assert all("tagged" not in alert for alert in first["alerts"])
    assert set(first["lines"][0]) == {"seq", "time", "tag", "known", "text"}

    monitor.step(30)
    second = monitor.feed(lines_after=2000, windows_after=len(first["windows"]), alerts_after=max(a["seq"] for a in first["alerts"]))

    assert [line["seq"] for line in second["lines"]] == list(range(2001, 2031))
    assert [w["number"] for w in second["windows"]] == list(range(len(first["windows"]) + 1, monitor.counts["windows"] + 1))
    assert monitor.feed(lines_after=2030, windows_after=monitor.counts["windows"], alerts_after=monitor.incident_seq)["lines"] == []


def test_a_page_that_holds_an_earlier_replay_is_told_to_start_again():
    monitor = Monitor(**quiet())
    monitor.control("start")
    monitor.step(500)
    generation = monitor.feed()["generation"]
    monitor.control("restart")
    monitor.step(100)

    stale = monitor.feed(lines_after=500, windows_after=9, alerts_after=9, generation=generation)

    assert stale["reset"] is True and stale["generation"] == generation + 1
    assert stale["lines"][0]["seq"] == 1 and stale["counts"]["lines"] == 100
    assert monitor.feed(lines_after=100, generation=generation + 1)["reset"] is False


def test_the_page_may_be_opened_in_the_middle_of_a_replay():
    monitor = Monitor(**quiet())
    monitor.control("start")
    monitor.step(4500)
    late = monitor.feed()

    assert late["counts"]["lines"] == 4500 and len(late["windows"]) == late["counts"]["windows"]
    assert {a["id"] for a in late["alerts"]} == {a["id"] for a in monitor.incidents}


def test_only_the_newest_lines_are_kept():
    monitor = Monitor(**quiet())
    monitor.step(live_core.FEED_LINES + 500)

    assert len(monitor.lines) == live_core.FEED_LINES and monitor.lines[0]["seq"] == 501


# ------------------------------------------------------------------------------------------------ the web server

class Server:
    def __init__(self, monitor):
        self.monitor = monitor
        self.server = live.make_server(monitor, port=0)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()

    def ask(self, path, body=None, kind="application/json", extra_headers=None):
        data = body if isinstance(body, bytes) else None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": kind} if data is not None else {}
        headers.update(extra_headers or {})
        request = urllib.request.Request(self.url + path, data=data, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=10) as answer:
                return answer.status, answer.headers, answer.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()


def test_the_server_gives_the_page_the_feed_and_takes_commands():
    monitor = Monitor(**quiet())

    with Server(monitor) as server:
        status, headers, body = server.ask("/")
        assert status == 200 and headers["Content-Type"].startswith("text/html") and b"LogBERT live monitor" in body
        assert headers["Cache-Control"] == "no-store"

        status, _, body = server.ask("/api/feed")
        assert status == 200 and json.loads(body)["state"] == "ready"

        status, _, body = server.ask("/api/control", {"action": "start", "speed": 250})
        assert (status, json.loads(body)) == (200, {"state": "running", "speed": 250})

        monitor.step(300)
        status, _, body = server.ask("/api/feed?lines=100&windows=0&alerts=0&generation=%d" % monitor.generation)
        answer = json.loads(body)
        assert status == 200 and answer["lines"][0]["seq"] == 101 and answer["counts"]["lines"] == 300

        assert json.loads(server.ask("/api/control", {"action": "pause"})[2])["state"] == "paused"
        assert json.loads(server.ask("/api/control", {"action": "restart"})[2])["state"] == "running"
        assert json.loads(server.ask("/api/feed")[2])["counts"]["lines"] == 0


def test_the_server_refuses_what_it_does_not_understand():
    monitor = Monitor(**quiet())

    with Server(monitor) as server:
        assert server.ask("/nothing")[0] == 404
        assert server.ask("/api/control", {"action": "start"}, kind="text/plain")[0] == 415  # another site's page cannot send this
        assert server.ask("/api/control", b"not json")[0] == 400
        assert server.ask("/api/control", {"action": "explode"})[0] == 400
        assert server.ask("/api/control", {"speed": "fast"})[0] == 400
        assert server.ask("/api/feed?lines=abc")[0] == 400
        assert server.ask("/api/nothing", {"a": 1})[0] == 404
        assert monitor.state == "ready"


def test_the_clock_thread_reads_lines_at_the_speed_and_stops():
    monitor = Monitor(speed=2000, **quiet())
    monitor.control("start")
    stop = threading.Event()
    thread = threading.Thread(target=live.pace, args=(monitor, stop), daemon=True)
    thread.start()
    time.sleep(0.6)
    stop.set()
    thread.join(timeout=3)

    assert not thread.is_alive() and 300 < monitor.position < 3000


def test_a_busy_port_is_reported_with_the_way_out(capsys):
    monitor = Monitor(**quiet())
    taken = live.make_server(monitor, port=0)

    try:
        port = taken.server_address[1]
        monkey = pytest.MonkeyPatch()
        monkey.setattr(live, "Monitor", lambda *a, **k: monitor)

        try:
            assert live.main(["--port", str(port)]) == 1
        finally:
            monkey.undo()

        assert "--port 9000" in capsys.readouterr().err
    finally:
        taken.server_close()


# ------------------------------------------------------------------------------------------------ the page

def page_script():
    return re.search(r"<script>(.*?)</script>", PAGE.read_text(encoding="utf-8"), re.S).group(1)


def test_the_page_uses_only_elements_that_it_has():
    html = PAGE.read_text(encoding="utf-8")
    used = set(re.findall(r'\$\("([A-Za-z]+)"\)', page_script())) | set(re.findall(r'getElementById\("([A-Za-z]+)"\)', page_script()))
    defined = set(re.findall(r'\sid="([A-Za-z]+)"', html))

    assert used and used <= defined, sorted(used - defined)


def test_the_page_asks_only_for_the_addresses_and_fields_that_the_server_gives(finished):
    script = page_script()

    assert set(re.findall(r'"(/api/[a-z]+)', script)) == {"/api/feed", "/api/control", "/api/upload", "/api/source"}

    # every field that the script reads from an answer (f), a window (w), an alert (a), a reason (r) or a log line (l) is one that the server sends
    answer = finished.feed()
    alert = answer["alerts"][0]
    sent = {"f": answer, "w": answer["windows"][0], "a": alert, "r": alert["reasons"][0], "l": answer["lines"][0], "c": answer["counts"]}

    assert alert["state"] == "replay_ended" and "replay_ended_at" in alert and "end_of_log" in alert

    for name, document in sent.items():
        reads = set(re.findall(r"\b" + name + r"\.([a-z_]+)\b", script))

        if name in ("r", "c"):
            # r is also a rectangle of the canvas code, c also a canvas element: only the names that belong to a reason or to the counters are compared
            reads &= {"template", "example", "count", "new"} if name == "r" else {"lines", "alert_lines", "unknown_lines", "windows", "flagged", "tp", "fp", "tn", "fn"}

        assert reads and reads <= set(document), (name, sorted(reads - set(document)))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed: the script of the page is not parsed")
def test_the_script_of_the_page_is_valid_javascript(tmp_path):
    script = tmp_path / "live.js"
    script.write_text(page_script(), encoding="utf-8")
    checked = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True, timeout=60)

    assert checked.returncode == 0, checked.stderr


def test_the_page_writes_log_text_only_as_text():
    # A log line is data from outside: it must never be put into the page as HTML. The only uses of innerHTML give it a fixed text.
    script = page_script()
    uses = [line.strip() for line in script.splitlines() if "innerHTML" in line]
    fixed = re.compile(r"""innerHTML\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*;""")

    assert uses and all(fixed.search(line) for line in uses), [line for line in uses if not fixed.search(line)]
    assert "insertAdjacentHTML" not in script and "document.write" not in script and "outerHTML" not in script


# ------------------------------------------------------------------------------------------------ loading another dataset

def upload_text(lines=12):
    return "\n".join(
        f"KERNDTLB {1117838570 + n} 2005.06.03 R00 2005-06-03-15.42.50 R00 RAS KERNEL FATAL "
        "uploadsentinel unfamiliar zephyr quartz xenon violet cobalt"
        for n in range(lines)
    )


def test_a_new_log_is_scored_with_the_existing_model_and_its_own_counts():
    monitor = Monitor(**quiet())
    detectors, generation = monitor.detectors, monitor.generation
    monitor.step(200)
    monitor.load_log(upload_text(), "new-dataset.log")

    assert monitor.detectors is detectors and monitor.generation == generation + 1
    assert monitor.dataset == "new-dataset.log" and len(monitor.source) == 12
    assert monitor.state == "ready" and monitor.position == 0 and monitor.incidents == []
    monitor.run_all()
    assert monitor.counts["lines"] == 12 and monitor.counts["windows"] == monitor.total_windows
    assert monitor.counts["flagged"] > 0 and len(monitor.incidents) == monitor.counts["flagged"]
    assert monitor.feed()["dataset"] == "new-dataset.log"


@pytest.mark.parametrize("text", [
    "", "plain log without BGL header", "name,score\nx,1", "\n \n",
    upload_text(2).replace("1117838571", "1117838569"),
    upload_text(1).replace("1117838570", "99999999999999999999999"),
])
def test_bad_uploads_leave_the_current_replay_untouched(text):
    monitor = Monitor(**quiet())
    monitor.step(20)
    before = monitor.feed()
    with pytest.raises(ValueError):
        monitor.load_log(text, "bad.log")
    assert monitor.feed() == before


def test_a_file_can_be_chosen_at_startup_and_the_demo_can_be_restored(tmp_path):
    path = tmp_path / "another.log"
    path.write_text(upload_text(), encoding="utf-8")
    monitor = Monitor(log_file=path, **quiet())
    detectors = monitor.detectors
    assert len(monitor.source) == 12 and monitor.dataset == "another.log"
    monitor.load_demo()
    assert len(monitor.source) == 30000 and monitor.dataset == "bgl-replay.log"
    assert monitor.detectors is detectors and monitor.state == "ready"


def test_the_server_accepts_a_new_log_and_can_restore_the_demo():
    monitor = Monitor(**quiet())
    generation = monitor.generation
    with Server(monitor) as server:
        status, _, body = server.ask("/api/upload", upload_text().encode(), kind="application/octet-stream",
                                      extra_headers={"X-Log-Name": "fresh%20log.txt"})
        assert status == 200 and json.loads(body)["lines"] == 12
        answer = json.loads(server.ask(f"/api/feed?generation={generation}")[2])
        assert answer["reset"] is True and answer["dataset"] == "fresh log.txt" and answer["state"] == "ready"
        assert answer["counts"]["lines"] == 0
        status, _, body = server.ask("/api/source", {"source": "bundled"})
        assert status == 200 and json.loads(body)["lines"] == 30000
        assert monitor.dataset == "bgl-replay.log"


def test_the_server_rejects_invalid_uploads_without_replacing_the_log(monkeypatch):
    monitor = Monitor(**quiet())
    before = monitor.feed()
    with Server(monitor) as server:
        assert server.ask("/api/upload", b"bad", kind="text/plain")[0] == 415
        assert server.ask("/api/upload", b"bad", kind="application/octet-stream")[0] == 400
        assert server.ask("/api/upload", b"\xff", kind="application/octet-stream")[0] == 400
        assert server.ask("/api/upload", b"bad", kind="application/octet-stream", extra_headers={"Content-Length": "-1"})[0] == 400
        assert server.ask("/api/source", {"source": "other"})[0] == 400
        assert server.ask("/api/source", b"not json")[0] == 400
        assert server.ask("/api/source", b"bad", kind="text/plain")[0] == 415
        monkeypatch.setattr(live, "MAX_LOG_BYTES", 1)
        assert server.ask("/api/upload", b"ab", kind="application/octet-stream")[0] == 413
        assert monitor.feed() == before
