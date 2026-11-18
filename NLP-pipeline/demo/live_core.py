# The back end of the live monitor (live.py): the replay file of the pipeline is played into the same code that the cluster's processor runs, one
# line after the other, as if a machine were writing the log right now. There is no web code in this file, so the tests can run it without a browser
# and without waiting (tests/test_live_core.py).
#
# What happens, in order (it is what processor.py and engine.py do in the cluster):
#   1. A log line arrives. Its alert tag is cut off (the tag is the ground truth, which the model never sees), Drain gives it a template, and
#      the line goes into the sliding 5-minute windows (Windower).
#   2. When a window is complete, LogBERT looks at it. The window is FLAGGED when more than r of its keys are not among the model's g best guesses.
#   3. Every flagged window creates its own ALERT. It stays open throughout playback. At the end it is marked REPLAY ENDED, which says
#      that playback stopped; it does not claim that a problem was repaired.
#   4. Because this is a labelled log, every alert also shows what was TRUE: whether the lines of its windows carry alert tags. A real monitor
#      has no such tag; here it is how the page tells a real alert from a false alarm.
#
# The page asks for news with feed(): the lines, windows and alerts that are newer than what it already has.

import sys
import threading
import time
from collections import deque, namedtuple
from pathlib import Path

import numpy as np

from demo_core import Row, clock, locate, window_ranks

# The newest log lines that are kept for the page, and the most that one answer carries (a slow page skips older lines; windows and alerts are never skipped).
FEED_LINES = 4000
MAX_PER_ANSWER = 400

# The speeds that the page offers (log lines per second) and the one it starts with. 30,000 lines last about 4 minutes at 120 lines per second.
SPEEDS = (30, 60, 120, 300, 1000)
DEFAULT_SPEED = 120

# How many message types the page names as the reason of an alert.
TOP_REASONS = 3

# The lines that are scored in one go while the lock is held (the page can ask for news in between).
SLICE = 25

# One line of the source: when, the alert tag ("-" for none), the message text.
Source = namedtuple("Source", "ts tag content")


class Monitor:
    def __init__(self, bundle=None, speed=DEFAULT_SPEED, say=print, log_file=None):
        default, folders = locate()

        for folder in folders:
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))

        from engine import Detectors, Item, Windower

        self.Item, self.Windower = Item, Windower
        self.bundle = Path(bundle) if bundle else default
        started = time.time()
        self.detectors = Detectors.load(str(self.bundle))
        self.g, self.r = self.detectors.g, self.detectors.r
        self.known = set(self.detectors.logbert.vocab.keys)
        self.speed = speed
        self.lock = threading.RLock()
        self.generation = 0
        source = Path(log_file) if log_file else self.bundle / "bgl-replay.log"
        self.load_log(source.read_text(encoding="utf-8-sig"), source.name)
        say(f"LogBERT loaded: {len(self.detectors.logbert.vocab) - 4} log keys, g = {self.g}, r = {self.r}. {len(self.source)} lines and {self.total_windows} windows "
            f"to replay ({time.time() - started:.1f} s)")

    # Validate a new source before replacing the current replay. Only BGL records in timestamp order are accepted.
    # The saved parser, model and detection settings are reused. No uploaded file is written into the repository.
    def load_log(self, text, name="uploaded.log"):
        from logad.data.bgl import parse_bgl_line
        from logad.data.windows import sliding_windows

        source = []

        for number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):
            if not line.strip():
                continue
            parsed = parse_bgl_line(line)
            if parsed is None or not parsed.content:
                raise ValueError(f"Line {number} is not a complete BGL log record. Keep the BGL header and message fields.")
            if not 0 <= parsed.timestamp <= 253402300799:
                raise ValueError(f"Line {number} has an invalid Unix timestamp.")
            if source and parsed.timestamp < source[-1].ts:
                raise ValueError(f"Line {number} is earlier than the previous record. Put the logs in timestamp order.")
            source.append(Source(parsed.timestamp, parsed.alert_tag, parsed.content))

        if not source:
            raise ValueError("The file contains no BGL log records.")
        if (source[-1].ts - source[0].ts) // self.detectors.step_seconds > 1000000:
            raise ValueError("The log spans too much time for this live monitor. Load a shorter time range.")

        pairs, _ = sliding_windows(np.array([s.ts for s in source], dtype=np.int64), self.detectors.size_seconds, self.detectors.step_seconds)
        with self.lock:
            self.source = source
            self.dataset = Path(name).name or "uploaded.log"
            self.total_windows = len(pairs)
            self.reset()

    def load_demo(self):
        self.load_log((self.bundle / "bgl-replay.log").read_text(encoding="utf-8-sig"), "bgl-replay.log")

    # ---------------------------------------------------------------- the state of one replay

    # Back to the start: nothing has been read, nothing is shown. The page notices the new generation and clears itself.
    def reset(self):
        with self.lock:
            self.generation += 1
            self.state = "ready"
            self.carry = 0.0
            self.position = 0
            self.rows = []
            self.windower = self.Windower(self.detectors.size_seconds, self.detectors.step_seconds)
            self.flushed = False
            self.lines = deque(maxlen=FEED_LINES)
            self.line_seq = 0
            self.windows = []
            self.incidents = []
            self.incident_seq = 0
            self.open = None
            self.log_time = None
            self.counts = {"lines": 0, "alert_lines": 0, "unknown_lines": 0, "windows": 0, "flagged": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # ---------------------------------------------------------------- playing

    # start, pause, resume, restart; and a new speed (lines per second) with any of them.
    def control(self, action=None, speed=None):
        with self.lock:
            if speed is not None:
                self.speed = max(1, min(5000, int(speed)))

            if action == "restart":
                self.reset()
                self.state = "running"
            elif action in ("start", "resume") and self.state in ("ready", "paused"):
                self.state = "running"
            elif action == "pause" and self.state == "running":
                self.state = "paused"
            elif action not in (None, "start", "resume", "pause", "restart"):
                raise ValueError(f"unknown action: {action}")

            return self.state

    # Time has passed: read as many lines as the speed allows. Returns how many were read.
    def tick(self, seconds):
        with self.lock:
            if self.state != "running":
                self.carry = 0.0
                return 0

            # a pause of the program (a slow window) is made up for, but never by more than two seconds of lines
            self.carry = min(self.carry + seconds * self.speed, self.speed * 2.0)
            count = int(self.carry)
            self.carry -= count

        done = 0

        while count > 0:
            done += self.step(min(count, SLICE))
            count -= SLICE

        return done

    # Read the next `count` lines (and score the windows that they complete). The lock is held only for this slice.
    def step(self, count):
        with self.lock:
            done = 0

            while done < count and self.position < len(self.source):
                self._read_line()
                done += 1

            if self.position >= len(self.source) and not self.flushed:
                for window in self.windower.flush():
                    self._score(window)

                self.flushed = True
                self._end_replay()
                self.state = "finished"

            return done

    # Everything, as fast as possible (the tests, and a quick look at the final numbers).
    def run_all(self):
        while self.state != "finished":
            if self.state in ("ready", "paused"):
                self.state = "running"

            self.step(1000)

    # ---------------------------------------------------------------- one line

    def _read_line(self):
        source = self.source[self.position]
        self.position += 1
        info = self.detectors.classify(source.content)
        known = info.template_id in self.known
        is_alert = source.tag != "-"
        self.rows.append(Row(source.ts, source.tag, source.content, info.template_id, info.template))
        self.log_time = source.ts

        self.counts["lines"] += 1
        self.counts["alert_lines"] += int(is_alert)
        self.counts["unknown_lines"] += int(not known)

        self.line_seq += 1
        self.lines.append({"seq": self.line_seq, "time": clock(source.ts)[11:], "tag": source.tag if is_alert else "", "known": known, "text": source.content})

        _status, closed = self.windower.add(self.Item(len(self.rows) - 1, source.ts, info.template_id, int(is_alert)))

        for window in closed:
            self._score(window)

    # ---------------------------------------------------------------- one window

    def _score(self, window):
        offsets = [item.offset for item in window.items]
        ids = [self.rows[i].template_id for i in offsets]
        started = time.perf_counter()
        worst, ranks = window_ranks(self.detectors, ids)
        seconds = time.perf_counter() - started
        alerts = sum(1 for i in offsets if self.rows[i].tag != "-")
        flagged = int(worst > self.r)
        outcome = "found" if flagged and alerts else "missed" if alerts else "false alarm" if flagged else "normal"

        entry = {"number": len(self.windows) + 1, "start": clock(window.start)[:16], "lines": len(offsets), "alert_lines": alerts, "missed": int(worst),
                 "flagged": flagged, "unknown": sum(1 for t in ids if t not in self.known), "outcome": outcome, "ms": round(seconds * 1000, 1)}
        self.windows.append(entry)

        self.counts["windows"] += 1
        self.counts["flagged"] += flagged
        self.counts[{"found": "tp", "missed": "fn", "false alarm": "fp", "normal": "tn"}[outcome]] += 1
        self._alert(entry, window, offsets, ids, ranks)

        return entry

    # ---------------------------------------------------------------- the alerts

    def _touch(self, alert):
        self.incident_seq += 1
        alert["seq"] = self.incident_seq

    # Which message types the model could not guess in this window, the most frequent first.
    def _reasons(self, offsets, ids, ranks):
        found = {}

        for offset, key, rank in zip(offsets, ids, ranks, strict=True):
            if rank is not None and (rank >= self.g or key not in self.known):
                row = self.rows[offset]
                entry = found.setdefault(key, {"template": row.template, "example": row.content, "count": 0, "new": key not in self.known})
                entry["count"] += 1

        return sorted(found.values(), key=lambda e: -e["count"])[:TOP_REASONS]

    def _alert(self, entry, window, offsets, ids, ranks):
        if not entry["flagged"]:
            return

        tagged = sum(self.rows[i].tag != "-" for i in offsets)
        alert = {"id": len(self.incidents) + 1, "state": "firing", "windows": 1, "peak": entry["missed"],
                 "first_window": entry["number"], "last_window": entry["number"], "limit": self.r,
                 "started": clock(window.start)[:16], "ended": clock(window.end)[:16], "alert_lines": tagged,
                 "real": tagged > 0, "peak_lines": entry["lines"], "reasons": self._reasons(offsets, ids, ranks)}
        self.incidents.append(alert)
        self.open = alert
        self._touch(alert)

    # Playback ended. This is not evidence that an incident recovered.
    def _end_replay(self):
        for alert in self.incidents:
            alert["state"] = "replay_ended"
            alert["replay_ended_at"] = clock(self.log_time)[:16] if self.log_time is not None else ""
            alert["end_of_log"] = True
            self._touch(alert)
        self.open = None

    # ---------------------------------------------------------------- what the page gets

    # News for the page: the lines (after `lines_after`), the windows (after `windows_after` of them) and the alerts that changed (after
    # `alerts_after`). A page that still holds an earlier replay (another generation) is told to start from nothing.
    def feed(self, lines_after=0, windows_after=0, alerts_after=0, generation=None):
        with self.lock:
            reset = generation is not None and generation != self.generation

            if reset:
                lines_after = windows_after = alerts_after = 0

            wanted = min(self.line_seq - lines_after, len(self.lines), MAX_PER_ANSWER)
            lines = list(self.lines)[len(self.lines) - wanted:] if wanted > 0 else []
            windows = self.windows[windows_after:]
            alerts = [dict(a) for a in self.incidents if a["seq"] > alerts_after]

            return {"generation": self.generation, "reset": reset, "state": self.state, "speed": self.speed, "speeds": list(SPEEDS),
                    "g": self.g, "r": self.r, "firing": self.open is not None, "dataset": self.dataset,
                    "position": self.position, "total_lines": len(self.source), "total_windows": self.total_windows,
                    "log_time": clock(self.log_time) if self.log_time else "", "counts": dict(self.counts),
                    "lines": lines, "windows": windows, "alerts": alerts}
