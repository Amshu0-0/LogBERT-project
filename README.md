# LogBERT on BGL

This project trains and evaluates LogBERT on the BlueGene/L system log, then runs the saved model in a local dashboard or a cloud pipeline.

| Folder | What it contains |
|---|---|
| [`NLP/`](NLP/README.md) | Data download, preprocessing, training, evaluation, saved results, figures and Colab notebooks |
| [`NLP-pipeline/`](NLP-pipeline/README.md) | The local dashboard, log-file uploads, Kafka/MongoDB pipeline, Kubernetes manifests and AWS deployment |

Keep both folders side by side: the pipeline imports the NLP package.

## Get the code and install

These commands are for macOS or Linux with Python 3.10 or newer, Git and `make`. On a Mac, `xcode-select --install` installs the command line tools that include `make`.

```bash
git clone https://github.com/Amshu0-0/LogBERT-project.git "$HOME/LogBERT-project"
cd "$HOME/LogBERT-project"
python3 -m venv NLP/.venv
source NLP/.venv/bin/activate
```

If the code is already in that folder, start with the `cd` line. On a Linux CPU machine, you can install the CPU build of PyTorch before the next command: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

```bash
python -m pip install -e "./NLP[dev,demo,docs]" -r NLP-pipeline/requirements-dev.txt
```

In a new terminal, run `cd "$HOME/LogBERT-project"` and `source NLP/.venv/bin/activate` again. The commands below run from the top folder, with that environment active.

## Open the live dashboard

```bash
make -C NLP-pipeline live
```

Open <http://127.0.0.1:8765> and press **Start**. Leave the terminal running; Ctrl+C stops the server.

The dashboard uses the included parser, model and 30,000-line replay. It creates one alert for each flagged window. Alerts stay open during playback and show **Replay ended** when the file finishes.

Press **Load log file** to score another BGL-format `.log` or `.txt` file, up to 20 MB per upload. Keep the BGL header fields and timestamp order. The saved model scores the new file; **Use demo log** restores the included replay. The filename, line count, window count and results update for each file. Other log formats need an adapter and a model suitability check.

For a file chosen at startup:

```bash
make -C NLP-pipeline live LIVE_ARGS="--file /absolute/path/to/another.log"
```

## Run checks and the local replay

| Command, from the top folder | What it checks |
|---|---|
| `make -C NLP lint test` | Code checks and 101 NLP tests |
| `make -C NLP-pipeline verify test` | 13 static checks and the updated suite of 272 pipeline tests; optional tests may be skipped when their packages are absent |
| `make -C NLP-pipeline local-replay` | Runs the included log through the saved model |
| `make -C NLP-pipeline demo` | Opens the detailed window and pasted-log demo at <http://127.0.0.1:7860> |

For the included replay, expect 30,000 lines, 394 windows, 116 anomalous windows and 266 flagged windows: 112 true positives, 154 false positives, 124 true negatives and 4 false negatives. In the live dashboard, these make 266 alerts. These are replay-slice results; the full test-period results are in [`NLP/README.md`](NLP/README.md#results).

## Train or reproduce the research

The dashboard can use the included model immediately. To run the research stages yourself, follow the [NLP run guide](NLP/README.md#how-to-run-it-step-by-step).

For the full BGL CPU run:

```bash
make -C NLP data
make -C NLP all CONFIG=configs/bgl_cpu.yaml
```

The download is about 700 MB unpacked. The stages prepare the data, train on normal training windows, tune on validation, evaluate on test, and write the comparison and analysis under `NLP/outputs/bgl_cpu/`. Runtime depends on your machine. The NLP guide also covers the small smoke test, GPU training on Colab, figures and restoring the saved model for its own demo.

## Run the cloud pipeline

Follow [Part B of the pipeline guide](NLP-pipeline/README.md#part-b-on-aws) for AWS prerequisites, image builds, deployment, smoke tests, replay, monitoring and teardown. The local dashboard above runs on your computer; cloud deployment creates billable AWS resources.

Version 1.0.1 has passed local tests. The saved cluster evidence is from version 1.0.0; version 1.0.1 still needs a real cluster run. Keep the Terraform state and SSH key of an existing cluster with that cluster when moving local folders.
