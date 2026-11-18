# The demo website: LogBERT on BlueGene/L system logs, in a browser.
#
#   cd NLP-pipeline/demo
#   python3 app.py                  open http://127.0.0.1:7860   (it needs about 20 seconds to start: it scores 394 windows)
#   python3 app.py --share          the same, with a temporary public link (for people who are not on your computer)
#
# Two things to do on the page: look at the windows of the replay (the same 30,000 lines, the same windows and the same verdicts as the cluster run of
# the pipeline), and paste lines of your own into a normal window to see what LogBERT makes of them. It needs only the files of the repository
# (see demo_core.py) and the packages in requirements.txt.

import argparse

import gradio as gr
import pandas as pd

from demo_core import Demo

FILTERS = [("all", "All windows"), ("anomalous", "Anomalous in the ground truth"), ("flagged", "Flagged by LogBERT"),
           ("false alarm", "False alarms (flagged, but normal)"), ("missed", "Missed anomalies (anomalous, not flagged)")]

def how_to_read(demo, summary):
    return f"""
### How to read this page

**The model.** LogBERT reads the log as a sequence of *message types* (the templates that the Drain parser finds; the model knows {len(demo.known)} of them).
For a window of 5 minutes it hides half of the keys, guesses them from the keys around them, and does this again until every key was hidden once.
A key is **MISSED** when the true key is not among the model's `g` best guesses (`rank` 0 is its first guess). The window is **flagged** when more
than `r` keys are missed. Here `g = {demo.g}` and `r = {demo.r}`, tuned on validation data of the NLP project, and the model is the variant that checks every
key once (A+).

**The ground truth.** The first field of a BGL line is an alert tag (`-` for a normal line). A window counts as *anomalous* when at least one of its
lines has a tag. The model never sees the tags: they are cut off before the line is parsed.

**Unknown keys.** A message type that did not occur in the training data has no place in the model's vocabulary. It is read as an *unknown key*, which can never
be guessed, so it always counts as missed. In this replay {summary['unknown_lines']:,} of the {summary['lines']:,} lines ({summary['unknown_templates']} of the {summary['templates']} message types) have an unknown key,
and {summary['flagged_by_unknown']} of the {summary['flagged']} flagged windows hold more than `r` of them in one stretch, which is enough to flag a window whatever the other lines say
({summary['hits_by_unknown']} of the {summary['hits']} windows that were rightly flagged, and {summary['false_alarms_by_unknown']} of the {summary['false_alarms']} false alarms). Only
{summary['flagged'] - summary['flagged_by_unknown']} flags depend on the order of the keys. The column *flagged because* in the table says which kind each flag is.

**What the numbers are, and are not.** The page replays 30,000 consecutive lines of the test period, not the whole test period. LogBERT flags many
windows (high recall, low precision), so its F1 on this slice is not the F1 of the report (which is computed on all 5,742 windows of the test period).
The numbers here are a consistency check: they are the numbers that the cluster of the pipeline produced for the same file.

**Why one pasted line rarely changes the verdict.** One line can add one missed key at most, and a window is flagged only above `r`. A burst of
unusual lines changes it.
"""


def frame(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame({"": []})


def build(demo):
    summary = demo.summary()
    counts = {"all": summary["windows"], "anomalous": summary["anomalous"], "flagged": summary["flagged"], "false alarm": summary["false_alarms"], "missed": summary["missed"]}
    intro = (f"""# LogBERT on BlueGene/L system logs
LogBERT learns which *sequences of log message types* are normal, hides some of the keys of a 5-minute window, guesses them from the rest, and flags
the window when more than r = {demo.r} hidden keys are not among its {demo.g} best guesses. The alert tags of the log are the ground truth, and the model never sees them.

**The replay on this page:** {summary['lines']:,} consecutive lines of the BGL test period ({summary['alert_lines']:,} with an alert tag), cut into {summary['windows']} windows
of 5 minutes (a new one every minute). {summary['anomalous']} windows hold an alert line. LogBERT flagged {summary['flagged']}: it found {summary['hits']} of the {summary['anomalous']} anomalous windows,
missed {summary['missed']}, and raised {summary['false_alarms']} false alarms (precision {summary['precision']:.2f}, recall {summary['recall']:.2f}, F1 {summary['f1']:.2f}).

**What decides most flags:** {summary['flagged_by_unknown']} of the {summary['flagged']} flagged windows are flagged by *unknown keys* alone. LogBERT never saw their message types in training ({summary['unknown_lines']:,} of the {summary['lines']:,} lines have such a key),
such a key can never be guessed, and more than r of them in one stretch is enough. Only {summary['flagged'] - summary['flagged_by_unknown']} flags depend on the order of the keys (tab 3 explains).""")

    def choices_for(which):
        windows = demo.pick(which)
        return [(demo.label(w), w.number) for w in windows], windows

    first = next((w for w in demo.windows if w.alerts and w.flagged), demo.windows[0])
    references = demo.reference_windows()

    with gr.Blocks(title="LogBERT on BGL logs") as page:
        gr.Markdown(intro)

        with gr.Tabs():
            with gr.Tab("1. Look at the windows of the replay"):
                which = gr.Radio(choices=[(f"{label} ({counts[key]})", key) for key, label in FILTERS], value="all", label="Which windows?")
                overview = gr.Dataframe(value=frame(demo.table(demo.pick("all"))), interactive=False, wrap=True, label="The windows (scroll; click a column name to sort)")
                choices, _ = choices_for("all")
                chosen = gr.Dropdown(choices=choices, value=first.number, label="Open a window")
                text, lines = demo.detail(first.number)
                verdict = gr.Markdown(text)
                listing = gr.Dataframe(value=frame(lines), interactive=False, wrap=True, label="Its lines: the alert tag of the log (the ground truth, - is normal) and the rank that LogBERT gave to the true key of each line")

                def on_filter(key):
                    options, windows = choices_for(key)
                    return frame(demo.table(windows)), gr.Dropdown(choices=options, value=options[0][1] if options else None)

                def on_window(number):
                    if not number:
                        return "No window in this list.", frame([])

                    text, lines = demo.detail(int(number))
                    return text, frame(lines)

                which.change(on_filter, which, [overview, chosen])
                chosen.change(on_window, chosen, [verdict, listing])

            with gr.Tab("2. Paste lines of your own"):
                gr.Markdown("Paste one or several log lines: whole BGL lines (their alert tag is cut off first) or just the messages. They are put into the middle of a "
                            "normal window of the replay that LogBERT leaves alone, and LogBERT looks at the window again.")
                pasted = gr.Textbox(lines=5, label="Log line(s), one per line")
                examples = demo.examples()
                burst = "\n".join(examples[:4] * 5) if examples else ""
                gr.Examples(examples=[[line] for line in examples] + ([[burst]] if burst else []), inputs=[pasted],
                            label="Click one to fill the box: real lines of the replay (the last one is a burst of 20 lines)")
                reference = gr.Dropdown(choices=[(demo.label(w), w.number) for w in references], value=references[0].number if references else None,
                                        label="Put them into this normal window")
                ask = gr.Button("Ask LogBERT", variant="primary")
                answer = gr.Markdown()
                table = gr.Dataframe(interactive=False, wrap=True, visible=False, label="The window with your line(s): the alert tag and the rank that LogBERT gave to each key")

                def on_ask(text, number):
                    message, rows = demo.try_line(text, int(number) if number else None)
                    return message, gr.Dataframe(value=frame(rows), visible=bool(rows))

                ask.click(on_ask, [pasted, reference], [answer, table])

            with gr.Tab("3. How to read this"):
                gr.Markdown(how_to_read(demo, summary))

    return page


def main(argv=None):
    parser = argparse.ArgumentParser(description="The LogBERT demo website")
    parser.add_argument("--bundle", help="the folder with drain.pkl, model.pt, settings.json and bgl-replay.log (default: NLP-pipeline/bundle)")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve on the network")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="make a temporary public link")
    args = parser.parse_args(argv)

    print("Loading LogBERT and scoring the windows of the replay (about 20 seconds)...")
    demo = Demo(args.bundle)
    page = build(demo)
    page.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
