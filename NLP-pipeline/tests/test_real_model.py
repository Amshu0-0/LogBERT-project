# The engine with the real trained LogBERT of the NLP project (NLP/results/logbert/model.pt). Only Drain is replaced by a small fake that
# knows one made-up template text for every key of the model's vocabulary: these tests are about the model, and they must not depend on the
# bundle/ folder of this computer (conftest.py).

import json
import re
import shutil
import time

import numpy as np
import pytest

import engine
from engine import BundleError, Detectors, Engine, UNSEEN
from fakes import bgl_line, record
from logad.logbert.dataset import window_to_chunks
from logad.logbert.trainer import load_checkpoint
from sink import MemorySink

from conftest import REPO

FILES = {"model.pt": REPO / "NLP" / "results" / "logbert" / "model.pt"}

pytestmark = pytest.mark.skipif(not all(p.exists() for p in FILES.values()), reason="the NLP project's results are not in this checkout")

SETTINGS = {"logbert": {"g": 8, "r": 8, "mask_ratio": 0.5, "full_coverage": True, "seed": 1234}, "window": {"size_seconds": 300, "step_seconds": 60}}


# Knows the templates that it is given (one text per key of the model's vocabulary).
class FakeDrain:
    def __init__(self, texts):
        self.templates = dict(texts)
        self.known = {content_of(text): (i, text) for i, text in self.templates.items()}

    def match(self, content):
        found = self.known.get(content)
        return (None, self.mask(content)) if found is None else found

    @staticmethod
    def mask(content):
        return re.sub(r"\d+", "<*>", content)


def content_of(template):
    return template.replace("<*>", "42")


def template_texts():
    _model, vocab, _centre, _checkpoint = load_checkpoint(FILES["model.pt"], "cpu")

    return {int(key): f"made up template number {key} with <*> in it" for key in vocab.keys}


def make_bundle(directory):
    directory.mkdir(parents=True, exist_ok=True)

    for name, source in FILES.items():
        shutil.copy(source, directory / name)

    (directory / "settings.json").write_text(json.dumps(SETTINGS))
    (directory / "drain.pkl").write_bytes(b"placeholder: the test replaces TemplateExtractor.load")


@pytest.fixture(scope="module")
def detectors(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bundle")
    make_bundle(directory)
    texts = template_texts()

    patch = pytest.MonkeyPatch()
    patch.setattr(engine.TemplateExtractor, "load", staticmethod(lambda path: FakeDrain(texts)))

    try:
        yield Detectors.load(directory)
    finally:
        patch.undo()


def test_the_bundle_loads_with_the_settings_of_the_nlp_project(detectors):
    assert (detectors.g, detectors.r) == (8, 8)
    assert (detectors.size_seconds, detectors.step_seconds) == (300, 60)
    assert detectors.logbert.full_coverage is True and detectors.logbert.mask_ratio == 0.5 and detectors.logbert.seed == 1234
    assert len(detectors.logbert.vocab) == 176


def test_every_line_of_a_known_template_gets_that_templates_id_and_text(detectors):
    texts = template_texts()

    for template_id, text in sorted(texts.items()):
        line = detectors.classify(content_of(text))

        assert line.template_id == template_id and line.template == text

    assert len(texts) == len(detectors.logbert.vocab.keys) == 172


def test_a_line_that_matches_no_template_is_unseen_and_keeps_its_masked_text(detectors):
    line = detectors.classify("a message nobody has ever seen 12345 data TLB error interrupt")

    assert line.template_id == UNSEEN
    assert line.template == "a message nobody has ever seen <*> data TLB error interrupt"
    assert detectors.classify("a message nobody has ever seen 12345 data TLB error interrupt") is line


def test_a_window_is_scored_like_the_nlp_code_scores_one_window(detectors):
    detector = detectors.logbert
    rng = np.random.default_rng(3)

    for length in (1, 2, 5, 37, 300, 511, 512, 700, 1200):
        ids = rng.choice(detector.vocab.keys, size=length)
        score = detectors.score(ids)

        # the batch function of the evaluation, asked about this one window (its masks are seeded with index 0, as ours are)
        import pandas as pd

        windows = pd.DataFrame({"start_idx": [0], "end_idx": [length]})
        batch = detector.score_windows(ids, windows)
        assert score.missed == int(batch.max_misses(8)[0]), length
        assert score.flag == int(batch.predict(8, 8)[0]) and score.chunks == len(window_to_chunks(detector.vocab.encode(ids), 512))


def test_the_scoring_batch_is_small_and_does_not_change_the_answer(detectors):
    # A very busy window must not need more memory than the processor may use (processor.yaml): the batch is small ...
    assert detectors.logbert.batch_size == engine.SCORE_BATCH == 16

    # ... and the answer does not depend on it: the masks are seeded by the position of the chunk, not by the batch.
    ids = np.random.default_rng(8).choice(detectors.logbert.vocab.keys, size=2300)
    small = detectors.score(ids)
    detectors.logbert.batch_size = 128

    try:
        assert detectors.score(ids) == small
    finally:
        detectors.logbert.batch_size = engine.SCORE_BATCH


def test_the_same_window_always_gets_the_same_answer(detectors):
    ids = np.random.default_rng(4).choice(detectors.logbert.vocab.keys, size=250)
    first = detectors.score(ids)

    for _ in range(3):
        detectors.score(np.random.default_rng(5).choice(detectors.logbert.vocab.keys, size=60))
        assert detectors.score(ids) == first


def test_unseen_templates_always_count_as_missed_keys(detectors):
    assert detectors.score([UNSEEN] * 20).missed == 20 and detectors.score([UNSEEN] * 20).flag == 1
    assert detectors.score([UNSEEN] * 3).missed == 3 and detectors.score([UNSEEN] * 3).flag == 0
    assert detectors.score([]) == engine.Score(0, 0, 0)


def test_the_longest_windows_of_the_log_are_scored_in_time(detectors):
    ids = np.random.default_rng(6).choice(detectors.logbert.vocab.keys, size=1500)
    started = time.perf_counter()
    score = detectors.score(ids)

    assert score.chunks == 3 and time.perf_counter() - started < 5


def test_a_whole_replay_with_the_real_detectors(detectors):
    texts = detectors.extractor.templates
    keys = detectors.logbert.vocab.keys
    rng = np.random.default_rng(7)
    ts, records = 1_118_000_000, []

    for offset in range(1500):
        ts += int(rng.integers(0, 3))
        tag = "KERNDTLB" if offset in (400, 401, 1200) else "-"
        line = bgl_line(ts, 0, tag).split(" t0")[0] + " " + content_of(texts[int(rng.choice(keys))])
        records.append(record(offset, {"kind": "log", "run": "demo", "seq": offset, "line": line}))

    records.append(record(1500, {"kind": "end", "run": "demo", "lines": 1500}))
    sink, machine = MemorySink(), Engine(detectors, MemorySink().load_state, lambda: 5.0)

    for start in range(0, len(records), 200):
        sink.write(machine.process(records[start: start + 200]))

    assert len(sink.events) == 1500 and len(sink.windows) > 20
    assert sink.runs["demo"]["status"] == "complete"
    assert sum(w["n_messages"] for w in sink.windows.values()) >= 1500
    assert all(w["logbert_chunks"] >= 1 and w["score_ms"] >= 0 for w in sink.windows.values())
    assert {w["truth"] for w in sink.windows.values()} == {0, 1}

    for event in sink.events.values():
        assert event["template_id"] in texts and event["truth"] in ("-", "KERNDTLB")


def test_the_bundle_must_belong_together(tmp_path):
    texts = template_texts()
    patch = pytest.MonkeyPatch()
    patch.setattr(engine.TemplateExtractor, "load", staticmethod(lambda path: FakeDrain(texts)))

    try:
        # a drain.pkl that lacks keys that the model knows (it comes from another preprocessing run)
        make_bundle(tmp_path / "wrong")
        some = sorted(texts)[:3]
        fewer = {i: t for i, t in texts.items() if i not in some}
        patch.setattr(engine.TemplateExtractor, "load", staticmethod(lambda path: FakeDrain(fewer)))

        with pytest.raises(BundleError, match=r"does not belong to the model: 3 keys of the LogBERT vocabulary"):
            Detectors.load(tmp_path / "wrong")

        # the right one loads
        patch.setattr(engine.TemplateExtractor, "load", staticmethod(lambda path: FakeDrain(texts)))
        assert Detectors.load(tmp_path / "wrong").g == 8
    finally:
        patch.undo()


@pytest.mark.parametrize("name", ["settings.json", "drain.pkl", "model.pt"])
def test_a_missing_file_says_which_one_and_what_to_run(tmp_path, monkeypatch, name):
    monkeypatch.setattr(engine.TemplateExtractor, "load", staticmethod(lambda path: FakeDrain(template_texts())))
    make_bundle(tmp_path / "b")
    (tmp_path / "b" / name).unlink()

    with pytest.raises(BundleError, match=name.replace(".", r"\.") + r" is missing.*export-nlp-bundle"):
        Detectors.load(tmp_path / "b")
