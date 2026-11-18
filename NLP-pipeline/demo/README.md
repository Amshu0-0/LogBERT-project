# The demos: LogBERT on BGL logs, in a browser

The command examples assume the repository is at `~/LogBERT-project` and its `NLP/.venv` environment is active.

Two small web pages run the NLP project's LogBERT. By default they use the replay file of the pipeline: the same 30,000 consecutive lines of the BGL test period, the same 5-minute windows and the same verdicts as the cluster run (394 windows, 116 anomalous, 266 flagged; `tests/test_demo_core.py` and `tests/test_live_core.py` compare them with `evidence/local-replay.json`). Neither needs a cluster, Kafka, MongoDB or a download: everything is in the repository (`bundle/`, `processor/engine.py`, `../NLP/src`).

| | Command | Address | What it is |
|---|---|---|---|
| **The live monitor** | `make live` | <http://127.0.0.1:8765> | The replay played as a live log, with an alert when LogBERT flags a window. It looks like a monitoring dashboard |
| **The demo website** | `make demo` | <http://127.0.0.1:7860> | Look at every window of the replay, and try log lines of your own |

For both, run the command in the folder `NLP-pipeline`, with the Python environment of the NLP project active (set-up: `README.md`, step A1).

## The live monitor

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make live
```

Open <http://127.0.0.1:8765> and press **Start**. It takes a few seconds to start, because it loads the model first.

**What happens.** The replay file is played line by line, like a log that a machine is writing right now. Every line goes through the same code as in the cluster (Drain finds the template, the line goes into the sliding windows). As soon as a 5-minute window is complete, LogBERT scores it. A window is **flagged** when more than 8 of its hidden message types are not among the model's 8 best guesses.

**Alerts.** Each flagged window creates one alert. It stays **Firing** throughout playback; unflagged windows do not close it. At the end of playback, every alert is labelled **Replay ended**. That does not claim that the anomaly was fixed. When an alert appears, the box *Alert state* turns red, a message pops up in the bottom corner, and the alert is added to the list on the right.

**Load another log.** Press **Load log file**, select a BGL-format `.log` or `.txt` file (up to 20 MB), and press **Start**. The filename and totals update to match the file. Blank lines are allowed; malformed records, empty files and records out of timestamp order are rejected with a message, leaving the current replay intact. The file stays in memory for this session. **Use demo log** restores the included replay.

The saved BGL parser and LogBERT model score the new file. A different log format needs an adapter and a separate check of whether this model is suitable. The BGL alert tags provide the ground truth shown by the page; they are not predictions. To load a file at startup:

```bash
make live LIVE_ARGS="--file /path/to/another.log"
```

**What you see on the page**

- *Alert state*, and the counters: log lines, windows scored, windows flagged, lines with an alert tag.
- *LogBERT score per window*: how many message types the model could not guess, one bar per window, with the limit as a dashed line. A bar above the line is a flagged window.
- *What was true*: how many lines each window held, and how many of them carry an alert tag of the dataset.
- *Alerts*: every alert, with its time, its windows, the three message types that the model could not guess (and whether it never saw them in training), and whether the alert was a **real anomaly** (the lines of its windows carry alert tags) or a **false alarm**. A real monitor has no such tags. They are used here only to say afterwards what was true; the model never sees them.
- *Live log*: the lines as they arrive. The coloured bar on the left of a line: green, LogBERT knows this message type; yellow, it never saw it in training; red, the line carries an alert tag. The switch *Only lines with an alert tag* filters them.
- Hover over a chart to see the numbers of one window.
- At the end, a summary: 266 of 394 windows flagged, 112 of the 116 anomalous windows found (recall 0.97), precision 0.42, and 266 alerts (112 real, 154 false alarms). These numbers describe the included demo file.

**Controls.** *Start*, *Pause*, *Restart*, and the speed (lines per second: 30, 60, 120, 300 or 1000; the whole replay takes about four minutes at 120). The switch *Sound* gives a short beep when an alert fires, and *Desktop alerts* shows a notification of your computer (the browser asks for permission first). Both are off at the start.

**Options**

```bash
make live LIVE_ARGS="--speed 300 --start"     # faster, and starts at once
make live LIVE_ARGS="--port 9000"             # another port, if 8765 is taken
```

Stop it with Ctrl+C. `--host 0.0.0.0` makes it open from other computers; there is no login, so everyone on the network can look. Without `make`: `python3 demo/live.py`.

## The demo website

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
pip install -r demo/requirements.txt
make demo
```

Open <http://127.0.0.1:7860>. It needs about 20 seconds to start, because it scores the 394 windows of the replay first. `make demo DEMO_ARGS=--share` also makes a temporary public link (it needs internet, the page still runs on your computer, and the link stops working when the program stops). Without `make`: `python3 demo/app.py` (options: `--port`, `--host 0.0.0.0`, `--share`, `--bundle FOLDER`).

1. **Look at the windows of the replay.** The numbers of the whole replay, a table of the windows (filter: all, anomalous in the ground truth, flagged by LogBERT, false alarms, missed anomalies), and a window opened in detail: every line with its alert tag (the ground truth, which the model never sees) and the rank that LogBERT gave to its true key (rank 0 = its first guess, `MISSED` = outside the best `g = 8`). The column *flagged because* says whether a window is flagged by unknown keys alone or because of the order of the keys.
2. **Paste lines of your own.** One or several lines (whole BGL lines lose their alert tag first), put into the middle of a normal window that LogBERT leaves alone. The page shows the verdict with and without them. One line can add one missed key at most, so a single line rarely flips a window; a burst of unusual lines does (the last example).
3. **How to read this.** The model, the ground truth, unknown keys, and what the numbers are and are not.

## What they show that the report's tables do not

In this replay 26,587 of the 30,000 lines have a message type that LogBERT never saw in training (68 of the 109 types in the file). Such a key can never be guessed, so it always counts as missed, and 258 of the 266 flagged windows hold more than `r = 8` of them in one stretch of 511 lines, which flags a window whatever the other lines say. Only 8 flags depend on the order of the keys. This is the same finding as the NLP project's analysis of unseen templates, seen window by window. (The pages compute these numbers from the files; they are not typed in.)

## Files

| File | What it is |
|---|---|
| `live.py` | The live monitor's small web server: the page, the news for the page (`/api/feed`, asked about three times a second) and the controls (`/api/control`) |
| `live_core.py` | What happens to the lines, without web code: the replay clock, the windows, LogBERT's verdicts, the alerts |
| `live.html` | The live monitor's page: one file, with its own styles and script, and no download from the internet |
| `app.py` | The demo website (Gradio): three tabs, no logic of its own |
| `demo_core.py` | The back end of the demo website, without web code: loads the bundle, cuts the replay into windows with the processor's `Windower`, scores them with LogBERT, explains a window, puts pasted lines into a window |
| `requirements.txt` | `pip install -r demo/requirements.txt`: Gradio, and the packages of the NLP project |

## What they do not do

They do not train anything and do not use the cluster. They show LogBERT only. The default 30,000-line replay has F1 0.59, compared with the report's F1 for LogBERT A+ of 0.305 on all 5,742 test windows. Loading another file in the live monitor gives that file's own results. Pasted lines (demo website) are judged inside a window of the replay, because LogBERT judges sequences, not single lines.

The alerts are created by this page: one per flagged window, kept open until playback ends. The cluster has no such alert rule, and none was added to Grafana. Sound and desktop notifications are optional browser features.
