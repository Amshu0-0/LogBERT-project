# The back end of the demo website: it reads the replay file of the pipeline, cuts it into the same 5-minute windows as the cluster, and lets
# LogBERT look at one window at a time. There is no web code in this file, so the tests can run it without a browser (tests/test_demo_core.py).
#
# What it needs (all of it is in the repository, nothing has to be downloaded):
#   - the folder bundle/ of the pipeline: drain.pkl (the log parser), model.pt (LogBERT), settings.json (g, r, the window) and bgl-replay.log
#     (30,000 consecutive lines of the BGL test period);
#   - the code of the processor (engine.py): the same file that runs in the cluster, so the windows and the verdicts are the same;
#   - the package logad of the NLP project (NLP/src), torch and numpy.
# They are found next to this folder (NLP-pipeline/bundle, NLP-pipeline/processor, NLP/src). live_core.py (the live monitor) uses the same files.

import time
from collections import namedtuple
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The template ID of a line that matches no known template (the same value as in the pipeline and in the NLP demo).
UNSEEN = -1

# At most this many lines of a window are listed in the table (the verdict is always about the whole window).
SHOWN_LINES = 300

# At most this many pasted lines are used.
MAX_PASTED = 200

# One line of the replay file: when, the alert tag (the ground truth, which the model never sees), the text, its template.
Row = namedtuple("Row", "ts tag content template_id template")

# One window of the replay: its number, the interval, the rows (indexes into Demo.rows), and what LogBERT said. unknown: how many of its lines have a
# key that LogBERT never saw in training (such a key can never be guessed, so it always counts as missed). forced: some chunk of the window (at most
# 511 lines, the length LogBERT reads at once) holds more than r such lines, so the window is flagged whatever the other lines say.
Win = namedtuple("Win", "number start end rows alerts flagged missed unknown forced")


# Where things are: the bundle folder, and the folders to put on the import path (the processor's engine.py and the NLP project's package logad).
def locate(here=HERE):
    pipeline = here.parent
    return pipeline / "bundle", [pipeline / "processor", pipeline.parent / "NLP" / "src"]


def clock(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))


# LogBERT on one window (the template IDs of its lines, in time order): the largest number of masked keys outside the top-g over the chunks of
# the window (the window is flagged when it is more than r), and the rank that LogBERT gave to the true key of every line (0 = its first guess;
# None where the line was not masked). It is the same calculation as Detectors.score() of the processor, plus the ranks. The caller has put the
# folders of locate() on the import path.
def window_ranks(detectors, template_ids):
    from logad.logbert.dataset import window_to_chunks

    logbert = detectors.logbert
    ids = np.asarray(template_ids, dtype=np.int64)
    ranks = [None] * len(ids)

    if len(ids) == 0:
        return 0, ranks

    body = logbert.model.max_len - 1
    chunks = window_to_chunks(logbert.vocab.encode(ids), logbert.model.max_len)
    worst = 0

    for c, prediction in enumerate(logbert.predict_sequences(chunks)):
        worst = max(worst, prediction.n_missed(detectors.g))

        for position, rank in zip(prediction.positions, prediction.ranks, strict=True):
            ranks[c * body + int(position) - 1] = int(rank)

    return worst, ranks


class Demo:
    def __init__(self, bundle=None, say=print):
        import sys

        default, folders = locate()

        for folder in folders:
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))

        from engine import Detectors, Item, Windower
        from logad.data.bgl import parse_bgl_line

        self.parse_bgl_line = parse_bgl_line
        self.bundle = Path(bundle) if bundle else default
        started = time.time()
        self.detectors = Detectors.load(str(self.bundle))
        self.g, self.r = self.detectors.g, self.detectors.r
        self.known = set(self.detectors.logbert.vocab.keys)
        self.unknown_rank = len(self.detectors.logbert.vocab)
        self.body = self.detectors.logbert.model.max_len - 1
        say(f"LogBERT loaded: {len(self.detectors.logbert.vocab) - 4} log keys, g = {self.g}, r = {self.r} ({time.time() - started:.1f} s)")

        # The lines, in file order, each with its template.
        self.rows = []
        windower = Windower(self.detectors.size_seconds, self.detectors.step_seconds)
        done = []

        for line in (self.bundle / "bgl-replay.log").read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = parse_bgl_line(line)

            if parsed is None:
                continue

            info = self.detectors.classify(parsed.content)
            self.rows.append(Row(parsed.timestamp, parsed.alert_tag, parsed.content, info.template_id, info.template))
            item = Item(len(self.rows) - 1, parsed.timestamp, info.template_id, int(parsed.is_anomalous))
            done.extend(windower.add(item)[1])

        done.extend(windower.flush())
        say(f"{len(self.rows)} lines read, {len(done)} windows of {self.detectors.size_seconds // 60} minutes (a new one every {self.detectors.step_seconds} s) ({time.time() - started:.1f} s)")

        # LogBERT on every window, as the processor does (one window per call).
        self.windows = []

        for number, window in enumerate(done, start=1):
            rows = [item.offset for item in window.items]
            ids = [self.rows[i].template_id for i in rows]
            worst, _ranks = self.misses(ids)
            alerts = sum(1 for i in rows if self.rows[i].tag != "-")
            unseen = [0 if i in self.known else 1 for i in ids]
            forced = max((sum(unseen[s:s + self.body]) for s in range(0, len(unseen), self.body)), default=0) > self.r
            self.windows.append(Win(number, window.start, window.end, rows, alerts, int(worst > self.r), worst, sum(unseen), forced))

        say(f"every window scored ({time.time() - started:.1f} s in all)")

    # ---------------------------------------------------------------- LogBERT on one window

    def misses(self, template_ids):
        return window_ranks(self.detectors, template_ids)

    # ---------------------------------------------------------------- the replay as a whole

    def summary(self):
        truth = [w.alerts > 0 for w in self.windows]
        flagged = [bool(w.flagged) for w in self.windows]
        hits = sum(t and f for t, f in zip(truth, flagged, strict=True))
        flags = sum(flagged)
        anomalous = sum(truth)
        precision = hits / flags if flags else 0.0
        recall = hits / anomalous if anomalous else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Flagged for that reason alone: a chunk with more unknown keys than r (each one counts as missed, whatever the lines around it are).
        by_unknown = [w for w in self.windows if w.flagged and w.forced]
        return {"lines": len(self.rows), "alert_lines": sum(1 for r in self.rows if r.tag != "-"), "windows": len(self.windows), "anomalous": anomalous,
                "flagged": flags, "hits": hits, "false_alarms": flags - hits, "missed": anomalous - hits,
                "flagged_by_unknown": len(by_unknown), "false_alarms_by_unknown": sum(1 for w in by_unknown if not w.alerts),
                "hits_by_unknown": sum(1 for w in by_unknown if w.alerts),
                "unknown_lines": sum(1 for r in self.rows if r.template_id not in self.known),
                "templates": len({r.template_id for r in self.rows}), "unknown_templates": len({r.template_id for r in self.rows if r.template_id not in self.known}),
                "precision": precision, "recall": recall, "f1": f1}

    # The outcome of one window against the ground truth.
    @staticmethod
    def outcome(window):
        if window.alerts and window.flagged:
            return "found"
        if window.alerts:
            return "missed"
        return "false alarm" if window.flagged else "normal, left alone"

    # The windows that the table shows: all, only the really anomalous, only the flagged, only the false alarms, only the missed ones.
    def pick(self, which="all"):
        keep = {"all": lambda w: True, "anomalous": lambda w: w.alerts > 0, "flagged": lambda w: w.flagged,
                "false alarm": lambda w: w.flagged and not w.alerts, "missed": lambda w: w.alerts and not w.flagged}[which]
        return [w for w in self.windows if keep(w)]

    @staticmethod
    def because(window):
        if not window.flagged:
            return ""

        return "unknown keys alone" if window.forced else "the order of the keys"

    def table(self, windows):
        return [{"window": w.number, "start (UTC)": clock(w.start), "lines": len(w.rows), "alert lines": w.alerts,
                 "LogBERT": "flagged" if w.flagged else "normal", "missed keys": w.missed, "unknown keys": w.unknown, "flagged because": self.because(w), "outcome": self.outcome(w)} for w in windows]

    def label(self, window):
        return (f"window {window.number} - {clock(window.start)[11:16]} - {len(window.rows)} lines - {window.alerts} alert lines - "
                f"LogBERT {'flagged' if window.flagged else 'normal'} ({window.missed} keys missed, {window.unknown} unknown)")

    # ---------------------------------------------------------------- one window in detail

    def status(self, rank):
        if rank is None:
            return "not masked"

        if rank >= self.unknown_rank:
            return "MISSED (key never seen in training)"

        return f"MISSED (rank {rank})" if rank >= self.g else f"ok (rank {rank})"

    # The table of lines: for every line the rank that LogBERT gave to its key (rank 0 = its first guess; MISSED = outside the top g guesses).
    def listing(self, rows, ranks, mine=()):
        table = []

        for position, (row, rank) in enumerate(zip(rows, ranks, strict=True)):
            template = row.template if row.template_id != UNSEEN else "(unseen) " + row.template
            table.append({"#": f"{position} (your line)" if position in mine else str(position), "tag": row.tag or "?",
                          "LogBERT": self.status(rank), "template": template, "message": row.content})

        return table

    def detail(self, number):
        window = self.windows[number - 1]
        rows = [self.rows[i] for i in window.rows]
        worst, ranks = self.misses([row.template_id for row in rows])
        assert worst == window.missed, "the same window must always give the same answer"
        lines = self.listing(rows, ranks)
        note = f" The table shows the first {SHOWN_LINES} of {len(lines)} lines." if len(lines) > SHOWN_LINES else ""
        text = (f"### Window {window.number}: LogBERT says **{'ANOMALOUS' if window.flagged else 'normal'}**, the ground truth is "
                f"**{'anomalous' if window.alerts else 'normal'}** ({self.outcome(window)})\n\n"
                f"{clock(window.start)} to {clock(window.end)} UTC, {len(window.rows)} lines, {window.alerts} of them with an alert tag. LogBERT hid half of the keys, "
                f"in turn, and could not find {worst} of them among its {self.g} best guesses; the window is flagged when that is more than r = {self.r}. "
                f"{window.unknown} of its lines have a key that LogBERT never saw in training: those can never be guessed."
                f"{f' This window is flagged by unknown keys alone: a stretch of up to {self.body} lines holds more than r of them, and each one counts as missed whatever the other lines say.' if window.flagged and window.forced else ''}{note}")
        return text, lines[:SHOWN_LINES]

    # ---------------------------------------------------------------- a line of your own

    # What the model will see of what was pasted: a whole BGL line loses its alert tag (kept as the ground truth); anything else is the message.
    def parse_input(self, text):
        text = text.strip()
        record = self.parse_bgl_line(text)
        content, tag = (record.content, record.alert_tag) if record else (text, None)
        template_id, template = self.detectors.extractor.match(content)
        return Row(None, tag, content, UNSEEN if template_id is None else template_id, template)

    # Windows to put a line into: no alert line, left alone by LogBERT, easy to read (30 to 150 lines), and with the fewest missed keys first, so that
    # the effect of what is pasted can be seen. If there are none like that, the nearest ones.
    def reference_windows(self, limit=30):
        quiet = [w for w in self.windows if not w.alerts and not w.flagged] or [w for w in self.windows if not w.alerts]
        readable = [w for w in quiet if 30 <= len(w.rows) <= 150] or [w for w in quiet if len(w.rows) >= 10] or quiet
        return sorted(readable, key=lambda w: (w.missed, -len(w.rows)))[:limit]

    # What the parser and the model know about a line.
    def knowledge(self, row):
        if row.template_id == UNSEEN:
            return "no template of the parser matches it: the model sees an unknown key, which can never be guessed"

        if row.template_id not in self.known:
            return "the parser knows this template, but LogBERT never saw it in training: an unknown key, which can never be guessed"

        return "LogBERT knows this key from training"

    # Put the pasted line(s), in the middle of a reference window, and look at the window with and without them. One line can add one missed key
    # at most, so a single line rarely changes the verdict; a burst of unusual lines does.
    def try_line(self, text, number=None):
        pasted = [line for line in (text or "").splitlines() if line.strip()][:MAX_PASTED]

        if not pasted:
            return "Paste a log line first.", []

        reference = self.windows[number - 1] if number else self.reference_windows()[0]
        mine = [self.parse_input(line) for line in pasted]
        rows = [self.rows[i] for i in reference.rows]
        before, _ = self.misses([row.template_id for row in rows])
        where = len(rows) // 2
        rows[where:where] = mine
        worst, ranks = self.misses([row.template_id for row in rows])
        positions = range(where, where + len(mine))
        mine_status = [self.status(ranks[i]) for i in positions]
        unseen = sum(1 for row in mine if row.template_id == UNSEEN)
        missed = sum(1 for s in mine_status if s.startswith("MISSED"))
        shown = ", ".join(f"`{row.content[:80]}`" for row in mine[:3]) + (" ..." if len(mine) > 3 else "")
        removed = [row.tag for row in mine if row.tag and row.tag != "-"]
        text = (f"**Model input ({len(mine)} line{'s' if len(mine) > 1 else ''}):** {shown}  \n"
                f"**Template of the first line:** `{mine[0].template}` - {self.knowledge(mine[0])}."
                f"{f' Alert tags removed before the model saw them: {len(removed)}.' if removed else ''}\n\n"
                f"### LogBERT says **{'ANOMALOUS' if worst > self.r else 'normal'}**\n"
                f"{worst} masked key(s) outside the top-{self.g} guesses (flagged if more than r = {self.r}). "
                f"Your line{'s' if len(mine) > 1 else ''}: {mine_status[0] if len(mine) == 1 else f'{missed} of {len(mine)} missed' + (f', {unseen} unseen' if unseen else '')}. "
                f"The same window without them: {before} missed ({'already flagged' if before > self.r else 'normal'}). Reference window: {self.label(reference)}.")
        return text, self.listing(rows, ranks, mine=set(positions))

    # A few real lines of the replay to try: the first alert lines with different templates, and a few normal ones.
    def examples(self, alerts=4, normal=2):
        seen, found = set(), {"alert": [], "normal": []}

        for row in self.rows:
            kind = "normal" if row.tag == "-" else "alert"

            if row.template_id in seen or len(found[kind]) >= (normal if kind == "normal" else alerts):
                continue

            seen.add(row.template_id)
            found[kind].append(row.content)

        return found["alert"] + found["normal"]
