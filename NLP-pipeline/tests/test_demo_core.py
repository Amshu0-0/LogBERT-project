# The demo website (demo/): its back end on the real bundle, and the page built around it. Building the back end scores the 394 windows of the replay
# (about 15 seconds), so it is done once for the whole file.
#
# The numbers must be the ones that the pipeline produced for the same file: evidence/local-replay.json (the same program as the cluster's, run on
# a Mac) and the cluster's own replay, which gave 394 windows, 116 anomalous, and 266 flagged by LogBERT.

import json
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE / "demo"))

from demo_core import SHOWN_LINES, UNSEEN, Demo  # noqa: E402

BUNDLE = PIPELINE / "bundle"

pytestmark = pytest.mark.skipif(not (BUNDLE / "model.pt").is_file(), reason="the bundle is not here: run make export-bundle")


@pytest.fixture(scope="module")
def demo():
    return Demo(say=lambda *_: None)


def quiet_window(demo):
    return demo.reference_windows()[0]


# ------------------------------------------------------------------------------------------------ the numbers

def test_the_replay_gives_the_numbers_of_the_pipeline_s_own_run(demo):
    saved = json.loads((PIPELINE / "evidence" / "local-replay.json").read_text(encoding="utf-8"))
    logbert = saved["detectors"]["logbert"]
    got = demo.summary()

    assert got["lines"] == saved["lines"]["stored"] == 30000 and got["alert_lines"] == saved["lines"]["with_alert_tag"]
    assert got["windows"] == saved["windows"] == 394 and got["anomalous"] == saved["anomalous_windows"] == 116
    assert (got["hits"], got["false_alarms"], got["missed"]) == (logbert["tp"], logbert["fp"], logbert["fn"])
    assert got["flagged"] == 266
    assert round(got["precision"], 4) == logbert["precision"] and round(got["recall"], 4) == logbert["recall"] and round(got["f1"], 4) == logbert["f1"]


def test_the_filters_add_up(demo):
    got = demo.summary()

    assert len(demo.pick("all")) == got["windows"]
    assert len(demo.pick("anomalous")) == got["anomalous"]
    assert len(demo.pick("flagged")) == got["flagged"]
    assert len(demo.pick("false alarm")) == got["false_alarms"]
    assert len(demo.pick("missed")) == got["missed"]
    assert {demo.outcome(w) for w in demo.windows} == {"found", "missed", "false alarm", "normal, left alone"}


def test_a_window_that_is_forced_by_unknown_keys_is_always_flagged(demo):
    # Every key is hidden once, an unknown key can never be guessed, so more than r of them in one chunk must flag the window.
    assert all(w.flagged for w in demo.windows if w.forced)
    assert all(w.missed >= w.unknown for w in demo.windows if len(w.rows) <= demo.body)

    got = demo.summary()
    assert got["flagged_by_unknown"] == sum(1 for w in demo.windows if w.flagged and w.forced) and got["flagged_by_unknown"] <= got["flagged"]
    assert got["hits_by_unknown"] + got["false_alarms_by_unknown"] == got["flagged_by_unknown"]


# ------------------------------------------------------------------------------------------------ one window

def test_a_window_is_scored_again_with_the_same_answer(demo):
    for number in (1, 3, 63):
        text, rows = demo.detail(number)  # detail() itself stops if the answer differs from the one of the table
        window = demo.windows[number - 1]

        assert f"Window {number}:" in text and len(rows) == min(len(window.rows), SHOWN_LINES)
        assert set(rows[0]) == {"#", "tag", "LogBERT", "template", "message"}


def test_a_long_window_shows_the_first_lines_and_says_so(demo):
    longest = max(demo.windows, key=lambda w: len(w.rows))
    assert len(longest.rows) > SHOWN_LINES

    text, rows = demo.detail(longest.number)
    assert len(rows) == SHOWN_LINES and f"first {SHOWN_LINES} of {len(longest.rows)}" in text


def test_a_key_that_the_model_never_saw_is_called_that(demo):
    unknown = next(w for w in demo.windows if w.unknown and len(w.rows) < 200)
    _, rows = demo.detail(unknown.number)

    assert any(row["LogBERT"] == "MISSED (key never seen in training)" for row in rows)
    assert all(row["LogBERT"] != "not masked" for row in rows)  # the variant of the model hides every key once


# ------------------------------------------------------------------------------------------------ lines of your own

def test_a_pasted_line_goes_into_the_middle_of_a_quiet_window(demo):
    reference = quiet_window(demo)
    assert not reference.alerts and not reference.flagged

    text, rows = demo.try_line(demo.examples()[0], reference.number)

    assert len(rows) == len(reference.rows) + 1
    assert [row["#"] for row in rows if "your line" in row["#"]] == [f"{len(reference.rows) // 2} (your line)"]
    assert "LogBERT says" in text and "The same window without them: 0 missed (normal)" in text


def test_one_line_adds_one_missed_key_at_most_and_a_burst_flags_the_window(demo):
    reference = quiet_window(demo)
    alert = demo.examples()[0]
    before = demo.misses([demo.rows[i].template_id for i in reference.rows])[0]

    one, _ = demo.try_line(alert, reference.number)
    missed = int(one.split("masked key(s)")[0].split()[-1])
    assert before <= missed <= before + 1 and "LogBERT says **normal**" in one

    burst, rows = demo.try_line("\n".join(demo.examples()[:4] * 5), reference.number)
    assert "LogBERT says **ANOMALOUS**" in burst and "Your lines: 20 of 20 missed" in burst
    assert sum("your line" in row["#"] for row in rows) == 20


def test_a_whole_bgl_line_loses_its_alert_tag(demo):
    row = next(r for r in demo.rows if r.tag != "-")
    line = f"{row.tag} {row.ts} 2005.11.25 R00-M0-N0 2005-11-25-09.43.19.000000 R00-M0-N0 RAS KERNEL FATAL {row.content}"

    parsed = demo.parse_input(line)
    assert parsed.tag == row.tag and parsed.content == row.content

    text, _ = demo.try_line(line, quiet_window(demo).number)
    assert "Alert tags removed before the model saw them: 1" in text


def test_a_blank_input_is_refused_and_a_message_that_no_template_matches_is_unknown(demo):
    assert demo.try_line("   \n  ", None) == ("Paste a log line first.", [])
    assert demo.try_line(None, None) == ("Paste a log line first.", [])

    parsed = demo.parse_input("zzqx completely new words nobody ever logged qqvw")
    assert parsed.tag is None and parsed.content.startswith("zzqx")

    if parsed.template_id == UNSEEN:
        assert "no template of the parser matches it" in demo.knowledge(parsed)
    else:
        assert parsed.template_id in demo.known or "never saw it in training" in demo.knowledge(parsed)


def test_at_most_two_hundred_pasted_lines_are_used(demo):
    text, rows = demo.try_line("\n".join(["rts panic! - stopping execution"] * 500), quiet_window(demo).number)

    assert sum("your line" in row["#"] for row in rows) == 200 and "Model input (200 lines)" in text


def test_the_reference_windows_are_quiet_and_start_with_the_clearest(demo):
    windows = demo.reference_windows()

    assert 1 <= len(windows) <= 30 and all(not w.alerts and not w.flagged for w in windows)
    assert [w.missed for w in windows] == sorted(w.missed for w in windows)


# ------------------------------------------------------------------------------------------------ the page

def test_the_page_is_built_from_the_back_end(demo):
    gradio = pytest.importorskip("gradio")
    import app

    page = app.build(demo)
    assert isinstance(page, gradio.Blocks)
