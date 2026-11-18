# NLP-pipeline: LogBERT on a cloud cluster

This folder takes the LogBERT model of the NLP project (`../NLP`) and runs it as a small cloud system. A replay file with 30,000 lines of the BlueGene/L supercomputer log (BGL) is sent through Kafka to a processor. The processor cuts the lines into 5-minute windows and asks LogBERT whether each window looks normal. The answers are stored in MongoDB and can be read through a REST API and a Grafana dashboard.

The cluster, the security and the monitoring are the ones from CA3. Terraform and Ansible build three AWS machines with Kubernetes (k3s). Kafka uses mutual TLS, MongoDB uses TLS, the REST API uses HTTPS and a login, network policies decide who may talk to whom, and Prometheus, Grafana and Loki watch everything.

```text
 replay file          producer Job            Kafka                  processor                      MongoDB          REST API
 bgl-replay.log  -->  "producer-replay"  -->  topic system-logs  -->  Drain finds the template  -->  events,    -->  /summary, /runs,
 (30,000 BGL lines)   (mutual TLS)            (one partition)         LogBERT checks every          windows,         /windows, /events
                                                                       5-minute window               runs             (HTTPS + login)
```

## Two ways to use it

| | Where it runs | What it costs | What you get |
|---|---|---|---|
| **A. On your computer** | Your laptop only | Nothing | The tests, the numbers of the replay, the live monitor, the demo website |
| **B. On AWS** | Three cloud machines | AWS charges while resources exist | The whole system: Kafka, MongoDB, the REST API and the dashboard |

Start with A. It needs nothing but Python, and it shows the same LogBERT numbers that the cluster has to reproduce.

## What you need

**For part A**

- Python 3.10 or newer, and `make`. On a Mac, `make` comes with the Xcode command line tools: `xcode-select --install`.
- This repository, with the folders `NLP` and `NLP-pipeline` side by side. The scripts look for `../NLP`.
- Disk space for the Python packages and included model (the PyTorch build is the largest part), and an internet connection for installation.
- No GPU, no Docker, no cloud account, and no download of the BGL log. The trained model and the replay file are already in `bundle/`.

**For part B, on top of that**

- Terraform 1.6 or newer (see `infra/terraform/versions.tf`), Ansible with `ansible-playbook`, `kubectl`, AWS CLI, and `ssh-keygen`.
- AWS credentials for the EC2/VPC resources in `infra/terraform/`, plus resource-tagging read access (`tag:GetResources`) for the cleanup scan. Configure the credentials with `aws configure`. The default region is `us-east-2`, set in `infra/config.yml`.
- If you already have this cluster in another local folder, keep its Terraform state and SSH key with it. A fresh clone does not contain those local files; do not create a second cluster to replace a missing state file.
- A GitHub account where Actions can run. GitHub builds the three container images, so you do not need Docker.

## Part A: on your computer

### A1. Set up Python (once)

The commands in this guide assume the repository is at `~/LogBERT-project`. If you chose another location, replace that part of each path. Each command block changes to its own working folder, so you can follow the sections in order.

```bash
cd "$HOME/LogBERT-project"
python3 -m venv NLP/.venv
source NLP/.venv/bin/activate
python -m pip install -e "./NLP[dev,demo,docs]" -r NLP-pipeline/requirements-dev.txt
```

On a Linux CPU machine, you can install the CPU build before the last command: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

This installs the NLP package, both demos, the plotting tools and the pipeline test dependencies in one environment. If you already created this environment, activate it and run the last command.

In each new terminal, activate the environment before using Python or `make`:

```bash
source "$HOME/LogBERT-project/NLP/.venv/bin/activate"
```

Check that it worked:

```bash
python3 -c "import torch, pandas, pyarrow, yaml, drain3, numpy, gradio; print('ok')"
```

It prints `ok`.

### A2. Run the tests

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make test
```

The updated suite contains 272 tests. Optional tests can be skipped if their packages are missing. The tests use stand-ins for Kafka and MongoDB, so nothing else has to run. The 34 live-monitor tests include file upload, invalid-file handling, one alert per flagged window, and the Replay ended status.

### A3. Replay the log on your computer

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make local-replay LOCAL_ARGS="--save local-replay.json"
```

This runs the processor's own code on the replay file, without Kafka and MongoDB, and prints a table. The LogBERT row must read:

```text
  detector                                  flagged     TP     FP     TN     FN  precision   recall      F1       FP rate
  LogBERT, full coverage (A+)                   266    112    154    124      4      0.421    0.966   0.586         55.4%
```

It also says that the replay has 30,000 lines (8,423 of them with an alert tag) and 394 windows (116 of them anomalous), and that these are exactly the windows the NLP code builds from the same file. It takes about ten seconds. The file `local-replay.json` is for comparing with the cluster later (part B); do not commit it.

These numbers are not the report's F1 of 0.305. The report scores all 5,742 windows of the test period; the replay file is a slice of 394 windows.

### A4. Watch it live: the live monitor

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make live
```

Open <http://127.0.0.1:8765> and press **Start**. The page plays the replay file line by line, like a log that a machine is writing right now. LogBERT scores every 5-minute window as soon as it is complete. When it flags a window, an **alert** fires: the "Alert state" box turns red, a message pops up in the bottom corner, and the alert is added to the list on the right.

The page shows:

- the log lines as they arrive (a green bar is a line that LogBERT knows, a yellow bar a message type it never saw in training, a red bar a line with an alert tag of the dataset);
- two charts, one for the LogBERT score of every window and one for what was really in it;
- every alert, with the three message types that LogBERT could not guess, and, because this is a labelled log, whether it was a **real anomaly** or a **false alarm**.

Each flagged window creates its own alert. Alerts stay open throughout playback, including across unflagged windows. When playback finishes, their status changes to **Replay ended**; that describes the end of the file, not recovery from a problem. For the included demo file, 266 of 394 windows are flagged and create 266 alerts (112 real, 154 false alarms).

To score another file, press **Load log file** and choose a BGL-format `.log` or `.txt` file (up to 20 MB per upload), then press **Start**. Keep the BGL header fields and put the records in timestamp order. The current filename, line count and window count change to match your file. The saved parser, LogBERT model and thresholds are reused. Results depend on the new logs, and ground-truth metrics depend on their BGL alert tags. Other log formats need an adapter and a model suitability check.

**Use demo log** restores the included 30,000-line file. You can also choose a file when starting the program:

```bash
make live LIVE_ARGS="--file /path/to/another.log"
```

Switch on **Sound** or **Desktop alerts** at the top right if you want a beep or a notification when an alert fires. The speed box sets how many log lines are played per second (the whole replay takes about four minutes at 120).

Options: `make live LIVE_ARGS="--speed 300 --start"` starts at once and faster. `LIVE_ARGS="--port 9000"` uses another port when 8765 is taken. Stop it with Ctrl+C. Nothing is installed or sent anywhere: the page and the program stay on your computer. More in `demo/README.md`.

### A5. Look at every window: the demo website

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make demo
```

Open <http://127.0.0.1:7860>. It needs about 20 seconds to start. The page has a table of all 394 windows of the replay, a detail view of one window (every line with the rank LogBERT gave it), and a box where you can paste log lines of your own. `make demo DEMO_ARGS=--share` makes a temporary public link (it needs internet). See `demo/README.md`.

### A6. Check that the files agree with each other

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make verify
```

It needs only `python3` with PyYAML and no cluster. It checks that the manifests, programs, ports, tags, secrets, network policies and saved evidence agree, and prints 13 checks that all say `PASS`.

### Only if you trained the model again: export the bundle

The repository already holds `bundle/`: the parser (`drain.pkl`), the trained LogBERT (`model.pt`), its settings (`settings.json`) and the replay file (`bgl-replay.log`). You only need this step if you train LogBERT again in the NLP project and want the pipeline to use the new model.

First make the processed data in `NLP` (`make data` downloads about 700 MB, then `make preprocess`; see `../NLP/README.md`). Then:

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make export-bundle
```

If the processed folder you have is not the one the trained model came from, the script stops with "does not belong to the model", names a folder that fits, prints the command to use, and leaves `bundle/` as it was. In that case run, for example:

```bash
make export-bundle BUNDLE_ARGS='--processed ../NLP/data/processed/bgl_memcheck'
```

After a new export, change `VERSION` (and the tag in the five manifests: `make images-info` lists them) so that the cluster does not keep an old image, then push to build new images (part B, B1).

## Part B: on AWS

Do these in order. Each step ends with a line that says whether it worked. **A running cluster costs money by the hour. Do B9 when you are done.**

### B1. Build the images (push, wait, make public)

The three images (`nlp-producer`, `nlp-processor`, `nlp-restapi`) are built by GitHub, because the model is baked into them. Push the repository:

```bash
cd "$HOME/LogBERT-project"
git add NLP-pipeline .github/workflows/nlp-pipeline-images.yml
git commit -m "NLP pipeline: images"
git push
```

On GitHub open *Actions*, then *NLP pipeline images*. The three jobs must end green. The workflow tries each image before it pushes it (the processor image loads the model, the producer image reads the replay file).

A package that is pushed for the first time is **private**, and the cluster pulls without a login. For each of the three packages: GitHub, your profile picture, *Your profile*, *Packages*, the package, *Package settings*, *Change visibility*, *Public*. Then:

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make check-images
```

Expected: `RESULT: PASS - all 3 images can be pulled without a login`.

The images are rebuilt when anything in `NLP-pipeline/producer`, `processor`, `restapi`, `bundle`, in `VERSION`, in `NLP/src` or in the workflow changes, and by hand (*Actions*, *Run workflow*). The image tag is the content of `VERSION`, and it is written in five manifests; `make verify` fails if they disagree.

### B2. Build the cluster

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make cluster-up
```

Terraform builds 19 AWS resources (network, firewall, key pair, three `t3.medium` machines) and Ansible installs Kubernetes (k3s) on them. Terraform downloads its own plug-ins the first time. It needs your AWS credentials (`aws configure` with the IAM user of the project). It takes a few minutes.

### B3. Start the stack

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make up
```

First it checks that the three images of this version can be pulled, and stops with the name of the one that cannot. Then it applies everything and waits until every part runs. The Job `bootstrap` makes the 13 passwords and certificates inside the cluster; nothing is made on your computer.

The processor needs a minute or two: it loads PyTorch and the model. It also waits for MongoDB and Kafka by itself, and makes the topic `system-logs` if it does not exist yet, so you do not have to do anything by hand while the stack starts. The two exporters may restart once or twice while Kafka and MongoDB start; that is normal. At the end it prints `Stack is up. Next:  make smoke ...`.

### B4. Send one line through everything

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make smoke SMOKE_ARGS="--save evidence/smoke-response.json"
```

A Job sends one tagged log line through producer, Kafka, processor, MongoDB and REST API. Expected: `RESULT: PASS - line smoke-... went producer -> Kafka ... -> MongoDB -> REST API in N s`. `make logs-trace TRACE=<id>` follows the same id through the logs of the programs.

### B5. Send the whole replay

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make replay RESULTS_ARGS="--windows 10 --compare local-replay.json"
```

The producer sends the 30,000 lines (about a minute), the processor scores the windows (a few minutes), and the command waits for it. It prints the LogBERT row, the first 10 windows that hold an alert line, and two checks: the cluster built exactly the windows that the NLP code expects, and the cluster and your computer (`local-replay.json` from A3) give the same numbers. LogBERT may differ by a window or two between a Mac (ARM) and the cluster (x86), because rounding can move a hidden key across the top-g line; the script says so and does not call that a failure.

It ends with `RESULT: PASS - the replay run-... is complete`. `make results` prints the same again at any time.

### B6. Run the checks

Run them after `make replay`, from the folder `NLP-pipeline`:

| Command | What it shows | Ends with |
|---|---|---|
| `make rest-check` | The REST API answers from your computer on all three machines, HTTPS only, login required | `RESULT: PASS - 3 of 3 nodes answered correctly ...` |
| `make metrics-check` | Prometheus, Loki and Grafana receive the numbers and the logs, and the dashboard is loaded | `RESULT: PASS - N of N monitoring checks passed` |
| `make netpol-check` | 169 real connections (13 kinds of pod, 13 targets) behave as the network policies say. The producer is a Job, so its 13 connections exist only during a replay; without a replay the count is 156 | `RESULT: PASS - all 169 connections behave as the policies say` |
| `make netpol-control` | The same test with the policies deleted, then put back: proves that the policies do the blocking | `RESULT: PASS - without the policies every connection worked ...` |
| `make heal-check` | Deletes a REST pod, MongoDB and Kafka, and stops and starts the processor, one at a time; each must come back and no data may be lost. Do not run it during a replay | `RESULT: PASS - 4 of 4 recovery tests passed` |
| `make exposure-check` | Only the intended ports of the machines answer from your computer | `RESULT: PASS - on all 3 nodes only ... answer` |
| `make rbac-check` | What each service account may and may not do | ten lines of yes and no |

### B7. Kill the processor during a replay (optional)

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make crash-test 2>&1 | tee evidence/crash-test.txt
```

This checks the restart design: if the processor dies in the middle of a replay, the finished result must be the same as without a crash. The program starts a replay, waits until 8,000 of the 30,000 lines are stored, kills the processor pod at once, waits for the new pod and for the end of the replay, and prints the table again. It takes about four minutes. Expected: the same LogBERT numbers as the normal replay and `RESULT: PASS - the processor was killed after N lines of run-...`. `CRASH_ARGS="--kill-after 12000"` kills later.

It works because the processor commits a Kafka offset only after its documents are written, and never past the oldest line that an open window still needs. Kafka hands those lines out again, the documents have fixed ids (a second write changes nothing), and LogBERT's random masks are seeded by the position of the chunk in the window, so a window scored again gets the same answer. `tests/test_restart.py` crashes the processor at every possible place against stand-ins and compares the result with an uninterrupted run.

### B8. Look at the dashboard

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make grafana-login
```

This prints the user name and the password (do not paste them anywhere). Then, in a second terminal window that stays open:

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make grafana
```

Open <http://localhost:3000>. What the dashboard shows is described below.

### B9. Take everything away

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
make teardown
```

It deletes the stack, destroys the machines and the network, and asks AWS whether anything tagged `Project=CS5287-NLP` or anything that costs money is left. It ends with `RESULT: CLEAN` or `RESULT: LEFTOVERS FOUND`. It does not touch the CA3 cluster, which has its own tag and its own Terraform state.

## Calling the REST API

The REST API is the one published part of the stack: HTTPS on port 30080 of any of the three machines, from the administrator's address only. Every address except the two probes needs a login.

| Endpoint | Answer |
|---|---|
| `GET /health`, `GET /ready` | Probes, no login. `/ready` says "ready" when MongoDB answers |
| `GET /stats` | How many documents each collection (`events`, `windows`, `runs`) holds |
| `GET /runs` | The replays, newest first, with status and number of lines and windows |
| `GET /summary?run=` | True and false positives and negatives, precision, recall, F1 and false-positive rate of LogBERT on the windows of a run, and LogBERT's time per window |
| `GET /windows?run=&flagged=&limit=&skip=` | The windows of a run in time order. `flagged=truth` or `flagged=logbert` keeps the windows where that verdict is 1 |
| `GET /events?run=&alert=1&limit=&skip=` | The log lines of a run in order. `alert=1` keeps the lines that carry an alert tag |
| `GET /alerts` | Up to 50 log lines with an alert tag, from the latest run |
| `GET /events/<trace id>` | The stored line of one smoke test (`404` with a JSON body when there is none) |

`run` is optional everywhere: without it the newest run is used. The certificate is signed by the private CA of the stack and has the name `restapi.nlp.test`, so a call to a machine's address uses `--resolve`. `make smoke` prints the exact command with the address filled in. `USER` and `PASSWORD` are the two values of the Secret `restapi-login`:

```bash
cd "$HOME/LogBERT-project/NLP-pipeline"
KUBECONFIG=.secrets/kubeconfig.yaml kubectl -n nlp get secret pki-ca -o jsonpath='{.data.ca\.crt}' | base64 --decode > ca.crt
curl -s --cacert ca.crt --resolve restapi.nlp.test:30080:<node-address> -u USER:PASSWORD https://restapi.nlp.test:30080/summary
```

`make rest-check` and `make results` read the login from the Secret themselves and never print the password. The REST API reaches MongoDB as an account that may only **read** the database `nlp`, so even a flaw in the program cannot write or delete data.

## What the dashboard shows

`make grafana` opens a dashboard of 27 charts in eight rows:

| Row | What is in it |
|---|---|
| *Right now* | Is a replay running, lines sent per second, lines waiting in Kafka, lines stored per second |
| *The replay (producer)* | Lines sent per second, how long the producer waits for Kafka |
| *Kafka and the processor* | Consumer lag, lines per second, the time from send to stored, how long the processor waits for MongoDB |
| *LogBERT* | Windows scored per minute, windows flagged per minute, LogBERT's seconds per window, *Lines that needed care* (no template known, too late for any window, out of order, delivered twice, not a valid message), log lines held in memory by open windows |
| *MongoDB* | Database writes per second, open connections |
| *REST API* | Requests per second, errors per second, how long an answer takes |
| *Is everything alive?* | Does Prometheus read every program, is the processor connected, seconds since the last stored line, producer send errors |
| *Logs (Loki)* | Log lines per second by program, errors and warnings of all programs, and *Follow one log line*: the box *Trace ID* at the top searches the logs of every program for one id |

During a replay the log's own clock runs faster than the wall clock: the producer sends 500 lines per second, while the real BGL log has fewer than one line per second on average. So a replay shows up as a burst, and the windows are scored much faster than they happened.

## Why it is built this way

| Choice | Reason |
|---|---|
| One topic partition, one processor pod, strategy `Recreate`, no autoscaler | A window is made of lines that follow each other in time. A partition keeps its order, one reader keeps one window in one place, and `Recreate` makes sure that there is never a second processor |
| Windows are cut by the **time of the log lines**, not by the clock | The NLP project evaluated windows of event time (5 minutes, sliding by 1 minute). The processor's `Windower` gives exactly those windows (`tests/test_windower.py` compares them with `logad.data.windows.sliding_windows` on random logs) |
| Offsets are committed by hand, after the documents are written; documents have fixed ids | A crash can repeat work but cannot lose or duplicate results |
| The model is baked into the processor image (`/bundle`) | The pod needs no volume and no download, and the image and the model cannot drift apart. The cost: a new export needs a new image |
| LogBERT scores a window in chunks of 16 masked copies (`SCORE_BATCH`) | The library's default (128) needed about 2 GB for a window of 40,000 lines; 16 keeps the memory flat |
| Lines whose template Drain does not know are counted, not dropped | They still count for LogBERT (as an unknown key), and `processor_unseen_lines_total` shows how many there were |
| Images are built by GitHub for amd64 only | The machines are amd64, and nothing here runs the images on a Mac |

## Security

Everything that CA3 had is still there, with the same manifests and the same code apart from names: mutual TLS for Kafka (the client certificate decides who may connect), TLS for MongoDB (`requireTLS`) with one account per program and the least rights each needs, HTTPS and a login for the REST API, 13 Secrets made inside the cluster by the Job `bootstrap` (no password or key on your computer or in the repository), Pod Security `restricted` on the namespace, 25 network policies (one that denies everything and 24 that allow the connections that are needed), and service accounts with the smallest roles (the processor and the producer cannot read Secrets; only `bootstrap` can create them). **The k3s setting that encrypts Secrets in the cluster database was not checked again for this cluster.**

What it does not cover:

- No rotation of passwords or certificates while the data stays; the only way to get new ones is `make down` and `make up`.
- No Kafka ACLs: every client with a certificate signed by the CA can use every topic (there is one).
- MongoDB clients use passwords, not certificates. One Secret value (the login of the MongoDB metrics reader) is in an environment variable; every other Secret is a read-only file.
- Monitoring traffic is plain HTTP inside the cluster. Grafana and Prometheus are not published; they are reached with `kubectl port-forward`.
- The root file system of the containers is writable, and there is no admission policy beyond Pod Security.
- The network policies trust labels: someone who can create pods in the namespace can give a pod a trusted label. Two policy files name the machines' subnet (`10.20.1.0/24`) and the address of the Kubernetes API Service (`10.43.0.1/32`), so they do not work in another network without a change.
- The live monitor and the demo website have no login. They listen on your computer only, unless you start them with `--host 0.0.0.0`; then anyone on the network can look.

## Known limitations

1. **The replay is a slice, not the whole test period.** It is 30,000 lines of a test period of millions. The numbers are a consistency check between your computer and the cluster. They are not a new evaluation of LogBERT, and they differ from the report's numbers.
2. **LogBERT in the cluster is not bit for bit the offline evaluation.** In the NLP project all chunks of all windows go through one call, and the random mask of a chunk is seeded by its number in that call. The processor scores one window per call, so a chunk is numbered inside its window and gets another random mask. Model, g, r, mask ratio and seed are the same, but some windows can get another verdict than in the report. The check that counts is the comparison of the cluster with the same program on your computer (`--compare`).
3. **One processor, no scaling.** The speed of a replay is set by LogBERT on two CPUs per pod and by `LINES_PER_SECOND` (500). A bigger replay file takes proportionally longer, and there is no way to add a second reader.
4. **One Kafka broker and one MongoDB, no replication.** Their disks are `local-path` volumes on one machine: a pod can be replaced without losing data, a machine cannot. `make heal-check` deletes pods, not machines.
5. **The start-up check of the bundle is lighter than before.** The processor checks that every key of the model exists in the parser. It cannot tell two runs of the same data apart. `make export-bundle` also compares the parser with `messages.parquet` and `templates.csv` when you export.
6. **The cluster side has run once, on 13 and 14 November 2026, with version 1.0.0.** See *What was tested, and where*. Version 1.0.1 (the one in this repository) has not run on a cluster.
7. **Terraform's state is local** (`infra/terraform/`, ignored by Git) and holds the administrator's address. If it is lost, `terraform destroy` cannot find the resources; the cleanup scan asks AWS and not the state file, so it still can.
8. **Only a Mac was the operator's computer.** Linux and Windows were not tried.

## What was tested, and where

| What | How | Where it was run |
|---|---|---|
| The programs: windows, restart after a crash at every point, the processor's start-up, sink, producer, REST API, result tables, export of the bundle, the image workflow, the dashboard, the live monitor and the demo | `make test`: 272 tests | a sandbox, with stand-ins for Kafka and MongoDB. **Run it on your computer too (A2)** |
| The files agree with each other (images and tags, ports, settings, Secrets, network policies against Services, Prometheus targets, rights, names used by the scripts, evidence free of addresses) | `make verify`: 13 checks, each shown to fail on deliberate mistakes | a sandbox, with a built-in manifest reader (no `kubectl` there). **Run it on your computer: it then uses `kubectl kustomize`** |
| LogBERT on the replay file | `make local-replay` | The author's Mac on 13 November 2026 (version 1.0.0), and a sandbox on 17 November (version 1.0.1): the same numbers, 266 windows flagged of 394 |
| The live monitor | `make live` | Opened locally on macOS on 18 November 2026. The updated upload and alert logic passed 34 tests on macOS and Linux. Sound and desktop notifications were not checked |
| Manifests against the Kubernetes schemas | `make validate` (needs `kubeconform`) | Not yet |
| Images, cluster and stack | `make check-images`, `make cluster-up`, `make up` | The author's Mac and AWS, 13 November 2026: three public packages, 19 resources, 3 nodes `Ready`; every part of the stack started except the processor, which stopped at its start-up (a mistake of version 1.0.0, fixed in 1.0.1) |
| Smoke test and replay | `make smoke`, `make replay` | The author's Mac and AWS, 13 and 14 November 2026: **passed**, twice. One line went through all five stages in 26 to 30 s. The replay (30,000 lines, 394 windows, 116 anomalous, 8,423 alert lines) gave the same numbers as `make local-replay`. LogBERT needed 64 to 72 ms per window on two CPUs |
| REST API, monitoring, healing, exposure, rights | `make rest-check`, `metrics-check`, `heal-check`, `exposure-check`, `rbac-check` | The same days: **all passed** (3 of 3 nodes; 4 of 4 monitoring checks; 4 of 4 recovery tests; only ports 22, 6443 and 30080 answer; ten rights as they must be) |
| Network policies | `make netpol-check`, `make netpol-control` | `make netpol-control` **passed** on 14 November: with the 25 policies a stranger pod was blocked from all 9 targets; with the policies deleted every connection was accepted within 1 s; with them applied again, the first answers were back within 2 s. `make netpol-check` **stopped** with `TypeError: 'NoneType' object is not iterable` (a Service without a pod has `endpoints: null`); fixed in the script with two tests, and **not run again** because the cluster had been taken down |
| Restart under a replay | `make crash-test` | 14 November 2026: **passed.** The processor was killed after it had stored 8,720 lines; the new pod was ready 18 s later and the finished replay gave the same numbers as one without a crash |
| Teardown | `make teardown` | 14 November 2026: **passed**, `RESULT: CLEAN` |

The saved output of the cluster runs is in `evidence/`. It was made with **version 1.0.0**, which also ran two older reference detectors (keyword rules and the saved decisions of a language model). Those files were not edited, so some of them still show the columns of the old detectors. See `evidence/README.md`.

**Version 1.0.1** has two changes, and **neither has run on a cluster**:

- The processor and the producer no longer stop at start-up for something that is not ready yet. Version 1.0.0 stopped when the topic `system-logs` did not exist yet (Kafka makes a topic when the first message arrives, and the check took the empty answer for a topic with the wrong number of partitions; the pod then restarted again and again). Now the processor waits for MongoDB and Kafka (it tries again after 2, 4 and 8 seconds, then every 15 seconds, for up to 20 minutes), makes the topic with one partition, and waits until Kafka lists it. A topic with more than one partition, or a wrong Kafka setting, still stops it, because waiting does not mend those. The producer waits for Kafka for up to 3 minutes and then stops, so that Kubernetes starts it again. `make up` also checks the images first. The tests go through `main()` with stand-ins (`tests/test_processor_main.py`, `tests/test_producer.py`); the ones about the missing topic fail on 1.0.0 and pass on 1.0.1.
- The two reference detectors are gone: the keyword rules, the saved decisions of the language model, and the code and files that belonged to them. The processor runs LogBERT only. The local replay gives the same LogBERT numbers before and after (394 windows, 116 anomalous, 266 flagged: 112 true and 154 false positives, 124 true and 4 false negatives).

**What no test can show:** that the real Kafka accepts the topic that the processor asks for, how long it takes to list it, and the real length of the waits. The first `make up` with 1.0.1 will show it.

## If something does not work

| If this happens | Do this |
|---|---|
| `make test` stops with `No module named ...` | The Python environment is not active or lacks a package. Run `source "$HOME/LogBERT-project/NLP/.venv/bin/activate"`, then `python -m pip install -r requirements-dev.txt` from `NLP-pipeline` |
| `make demo` says `No module named 'gradio'` | `pip install -r demo/requirements.txt`. `make live` does not need it |
| `make live` says `Cannot open port 8765` | Another program uses the port (maybe an older `make live` that is still running). Use another one: `make live LIVE_ARGS="--port 9000"` |
| The live monitor page says "The page cannot reach the program" | The program in the terminal stopped. Start `make live` again and reload the page |
| `make export-bundle` says `drain.pkl ... was not found` | The processed data is not on this computer. Run `make data` and `make preprocess` in `NLP` (or copy `NLP/data/processed/bgl` from the computer where you ran them) |
| `make export-bundle` stops with "does not belong to the model" | The processed folder is not the run that the trained files come from. The message names a folder that fits and prints the command; `bundle/` was not changed |
| `make export-bundle` prints `Note: for N of M lines ... the parser's template is not the one in messages.parquet` | Go on. It is the way Drain works: while `make preprocess` reads the log, Drain puts each line into the nearest group it knows, and the finished parser asks for an exact match. On the author's computer it was 3.3% of the lines |
| `make export-bundle` prints `WARNING: the templates of drain.pkl differ from templates.csv` | Do not go on: the two files come from different runs of `make preprocess`. Run it again, or copy both from the same run |
| The workflow stops with "a file of the bundle is missing" | Run `make export-bundle`, then `git add NLP-pipeline/bundle`, commit and push |
| `make check-images` says an image cannot be pulled, or `make up` stops at once with `RESULT: FAIL` and the name of an image | The package is still private, or the tag was not built yet. After a push, wait until the three jobs of the workflow are green, and make each package public once (B1). Then `make check-images` must say `PASS`. If your computer has no access to ghcr.io but the cluster has, `SKIP_IMAGE_CHECK=1 make up` skips the check |
| A pod is in `ImagePullBackOff` | The same: the package is private, or the tag in the manifest is not one that was pushed. `make verify` checks the tags against `VERSION` |
| The cluster runs an old model after you exported again | The image tag did not change, so a machine kept its copy. Change `VERSION` and the tag in the five manifests together (`make images-info` lists them), push, and wait for the build |
| `make up` stops with an error | Read the first error. Then `make diagnose` (read-only: states, events, logs). It is safe to run `make up` again |
| The processor is in `CrashLoopBackOff` and `make logs` shows `the topic must have exactly one partition` with `"partitions":0` | That is the mistake of version 1.0.0, and the cluster is still running the old images. Check that `VERSION` says 1.0.1 (or newer), that the workflow built it, and run `make up` again. To get an old cluster going without waiting, make the topic once by hand: `KUBECONFIG=.secrets/kubeconfig.yaml kubectl -n nlp exec kafka-0 -c kafka -- sh -c 'kafka-topics --bootstrap-server 127.0.0.1:9094 --create --if-not-exists --topic system-logs --partitions 1 --replication-factor 1'`, then `kubectl -n nlp delete pod -l app.kubernetes.io/name=processor` |
| `make logs` of the processor shows `waiting for MongoDB`, `waiting for Kafka, to make sure that the topic exists` or `waiting for the partitions of the topic` | Version 1.0.1 waits for the parts that are not ready yet and tries again by itself, for up to 20 minutes. That is normal in the first minutes of `make up`, and a line `... is ready` follows. If the same error comes back for more than 5 minutes, read `error` and `detail`, and run `make diagnose`. After 20 minutes the processor says `gave up waiting` and stops, and Kubernetes starts it again |
| The log of the producer Job shows `waiting for Kafka` | The producer could not reach Kafka at the first try and tries again, for up to 3 minutes. That is normal when Kafka has just started. After 3 minutes it stops and Kubernetes starts it again within the limits of the Job. If it keeps failing, `make diagnose` |
| The processor stops with `the topic must have exactly one partition` and a number of 2 or more | The topic was made with another number of partitions, and the windows need exactly one. `make down`, then `make up`, starts again without the old topic (all stored data is removed) |
| The processor pod is `Running` but not ready for minutes | It is loading PyTorch and the model; `make logs` shows it. If it is killed with `OOMKilled`, `make diagnose` says so: the memory limit in `processor/processor.yaml` is too low for the window that it was scoring |
| `make replay` says the producer Job did not finish | Its last log lines are printed above that message. Then `make diagnose` |
| `make replay` waits and then says the replay is not complete | The processor is behind or stopped. `make status` and `make logs`. The numbers shown are those of the windows scored so far; `make results` prints them again later |
| The comparison at the end of `make replay` shows a difference in the number of windows or lines | That must not happen: the same file gives the same windows. Keep the output, and check that `make local-replay` used the same `bundle/` that went into the images (the images are built from the repository; `bundle/` on your computer may be newer) |
| The comparison shows a difference of a window or two in the LogBERT row | Expected between a Mac and the cluster (B5) |
| `make scan` after `make cluster-down` finds something | It names it. Delete it in the AWS console, or run `make scan SCAN_ARGS="--all-regions"` if it may be in another region |
