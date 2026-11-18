# This program reads BGL system-log lines from Kafka, runs the NLP project's LogBERT on them, and saves the results into MongoDB.
# It replaces the CA3 processor, which stored network flows: the pipeline around it (Kafka, MongoDB, TLS, policies, monitoring) is the same.
#
# What it does with the messages is in engine.py (Drain, LogBERT and the windows) and sink.py (the writing). This file is the part that
# lives in a Kubernetes pod, and it is the CA3 processor with these changes:
#   - a message is a log line, not a flow. Every line becomes one document ("events"); every 5-minute window becomes one document
#     ("windows"), with LogBERT's verdict; every replay becomes one document ("runs").
#   - the Kafka offsets are committed by hand, AFTER the documents are written, and not past the oldest line that an open window still
#     needs. If the pod dies, Kafka hands those lines out again, and the same documents are written again (they have fixed ids).
#   - the topic must have ONE partition: a window is made of lines that follow each other in time, and a partition keeps its order.
#     (The CA3 processor read 3 partitions with several replicas; this one is a single replica on purpose, see processor.yaml.)
#   - the start-up WAITS instead of stopping. Kafka, the topic and MongoDB may not be ready when the pod starts (a cold cluster needs
#     minutes), so the pod waits for each of them, and its log says which one. A topic that does not exist yet is made by the processor
#     itself, with ONE partition. (Version 1.0.0 stopped at the first try: a topic that Kafka had not made yet sent the pod into
#     CrashLoopBackOff every time, and the stack came up only after a command by hand.)
#
# What is kept from CA3:
#   1. The MongoDB credentials come from files that Kubernetes mounts from a Secret (MONGO_USER_FILE, MONGO_PASSWORD_FILE).
#   2. A small HTTP server answers /healthz, /readyz and /metrics (Kubernetes probes and Prometheus).
#   3. SIGTERM stops the program cleanly: it finishes the batch it is working on.
#   4. Both connections are encrypted: Kafka with TLS and a client certificate (mutual TLS), MongoDB with TLS and a user that may only
#      read and write its own database. There is no way to turn TLS off.
#   5. Every log line of the program is one JSON object, for Grafana Alloy and Loki.
#   6. If a batch cannot be stored, the program logs the error and stops WITHOUT committing; Kubernetes restarts it.

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

SERVICE = "processor"

# If the loop has not asked Kafka for this many seconds, /healthz says it is stuck and Kubernetes restarts the container.
STALL_SECONDS = 120

# Lines per batch. A batch is written to MongoDB with three bulk calls and then committed.
MAX_RECORDS = 500

# How long the start-up waits for MongoDB, Kafka and the topic before it gives up and stops (Kubernetes then starts the pod again).
# START_WAIT_SECONDS in the environment changes it. A cold cluster needs a few minutes: Kafka and MongoDB start at the same time as this
# pod. While it waits, the pod is Running but not ready, its log says what it waits for, and /healthz still says ok (the loop has not
# started, so nothing can be stuck).
START_WAIT_SECONDS = 1200


# One log line = one JSON object on standard output.
def log(level, message, **fields):
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "level": level, "service": SERVICE, "msg": message}
    record.update(fields)
    print(json.dumps(record, separators=(",", ":"), default=str), flush=True)


# One required setting from an environment variable. No default: a missing setting stops the program and names it.
def require(name):
    value = os.environ.get(name)

    if not value:
        sys.exit(f"missing required setting: {name}")

    return value


# One required credential: NAME_FILE (a mounted Secret) or, for a quick local run, NAME.
def require_secret(name):
    path = os.environ.get(name + "_FILE")

    if path:
        try:
            with open(path) as f:
                value = f.read().rstrip("\n")
        except OSError as error:
            sys.exit(f"cannot read {name}_FILE ({path}): {error}")

        if not value:
            sys.exit(f"{name}_FILE ({path}) is empty")

        return value

    value = os.environ.get(name)

    if not value:
        sys.exit(f"missing required setting: {name} (or {name}_FILE)")

    return value


# Every setting, read before anything is connected, so a missing one is reported first.
class Settings:
    def __init__(self):
        self.broker = require("KAFKA_BROKER")
        self.topic = require("KAFKA_TOPIC")
        self.group = require("CONSUMER_GROUP")
        self.ssl_ca = require("KAFKA_SSL_CA_FILE")
        self.ssl_cert = require("KAFKA_SSL_CERT_FILE")
        self.ssl_key = require("KAFKA_SSL_KEY_FILE")
        self.mongo_tls_ca = require("MONGO_TLS_CA_FILE")
        self.mongo_host = require("MONGO_HOST")
        self.mongo_port = int(require("MONGO_PORT"))
        self.database = require("MONGO_DB")
        self.events = require("MONGO_COLLECTION")
        self.windows = require("MONGO_WINDOWS_COLLECTION")
        self.runs = require("MONGO_RUNS_COLLECTION")
        self.metrics_port = int(require("METRICS_PORT"))

        # The folder with drain.pkl, model.pt and settings.json (inside the image).
        self.bundle = require("BUNDLE_DIR")

        # How many CPU threads PyTorch may use. The nodes have 2 vCPUs and share them with Kafka and MongoDB.
        self.torch_threads = int(os.environ.get("TORCH_THREADS", "2"))

        # How long the start-up waits for MongoDB, Kafka and the topic, in seconds.
        self.start_wait_seconds = float(os.environ.get("START_WAIT_SECONDS", str(START_WAIT_SECONDS)))

        self.mongo_user = require_secret("MONGO_USER")
        self.mongo_password = require_secret("MONGO_PASSWORD")


# ---------------------------------------------------------------- the numbers (served at /metrics)

MESSAGES = Counter("processor_messages_total", "Log lines read from Kafka and stored in MongoDB.")
ALERTS = Counter("processor_alerts_total", "Stored log lines that carry a BGL alert tag (the ground truth of the dataset).")
BATCHES = Counter("processor_insert_batches_total", "Batches written to MongoDB (events, windows and runs).")
WINDOWS = Counter("processor_windows_total", "Windows scored and stored.")
FLAGGED = Counter("processor_windows_flagged_total", "Windows that LogBERT flagged as anomalous.", ["detector"])
UNSEEN = Counter("processor_unseen_lines_total", "Lines that match no template of the trained Drain parser.")
LATE = Counter("processor_late_lines_total", "Lines that came too late for any open window (not windowed).")
REORDERED = Counter("processor_reordered_lines_total", "Lines that arrived after a newer line (still windowed).")
REDELIVERED = Counter("processor_redelivered_lines_total", "Messages that Kafka handed out again and the processor skipped.")
BAD = Counter("processor_bad_messages_total", "Messages that were not a valid log line or marker and were skipped.")

INSERT_SECONDS = Histogram("processor_insert_batch_seconds", "Seconds that the three writes of one batch took: the time spent waiting for MongoDB.",
                           buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
WINDOW_SECONDS = Histogram("processor_window_score_seconds", "Seconds that LogBERT needed for one window.",
                           buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
END_TO_END = Histogram("processor_end_to_end_seconds", "Seconds from the producer's send (the Kafka timestamp of the message) to the moment the line is stored in MongoDB.",
                       buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600))

READY = Gauge("processor_ready", "1 when the processor is connected to Kafka and MongoDB and has loaded the models.")
LAST_MESSAGE = Gauge("processor_last_message_timestamp_seconds", "Unix time of the last stored log line (0 if none yet).")
OPEN_LINES = Gauge("processor_open_window_lines", "Log lines held in memory because an open window still needs them.")


# The facts that the probes need. The loop writes them, the HTTP server thread reads them.
class State:
    def __init__(self):
        self.ready = False
        self.looping = False
        self.last_poll_time = time.time()


state = State()


def set_ready(value):
    state.ready = value
    READY.set(1 if value else 0)


set_ready(False)
LAST_MESSAGE.set(0)


# The three small pages Kubernetes and Prometheus ask for.
class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, content_type="text/plain; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Liveness: is the loop still turning? (Not judged while the models load: the startup probe covers that.)
        if self.path == "/healthz":
            stalled = state.looping and time.time() - state.last_poll_time > STALL_SECONDS
            self.reply(503 if stalled else 200, "consumer loop stalled\n" if stalled else "ok\n")

        # Readiness: connected to Kafka and MongoDB, models loaded?
        elif self.path == "/readyz":
            self.reply(200 if state.ready else 503, "ready\n" if state.ready else "not ready\n")

        elif self.path == "/metrics":
            self.reply(200, generate_latest(), CONTENT_TYPE_LATEST)

        else:
            self.reply(404, "not found\n")

    def log_message(self, format, *args):
        pass


def start_health_server(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------- Kafka

# A message that is not valid JSON becomes None; the engine counts it as bad and skips it (one bad message must not stop the pipeline).
def read_json(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        return None


# The offset to commit, as the installed kafka-python wants it (2.0 has two fields, later versions have a third).
def offset_and_metadata(offset):
    from kafka.structs import OffsetAndMetadata

    try:
        return OffsetAndMetadata(offset, "", -1)
    except TypeError:
        return OffsetAndMetadata(offset, "")


def commit(consumer, topic, offsets):
    from kafka.structs import TopicPartition

    consumer.commit({TopicPartition(topic, partition): offset_and_metadata(offset) for partition, offset in offsets.items()})


# ---------------------------------------------------------------- the loop

# Read, process, write, commit, again and again. `stop()` says when to leave (after the batch that is running).
def serve(consumer, engine, sink, topic, stop, max_records=MAX_RECORDS):
    from engine import Record

    count = logged = alerts_total = windows_total = 0
    partitions = set()
    state.looping = True

    while not stop():
        batches = consumer.poll(timeout_ms=1000, max_records=max_records)
        state.last_poll_time = time.time()
        messages = [message for records in batches.values() for message in records]

        if not messages:
            continue

        partitions.update(message.partition for message in messages)

        if len(partitions) > 1:
            raise RuntimeError("the topic has more than one partition: the windows need the lines in order, so it must have exactly one")

        records = [Record(m.topic, m.partition, m.offset, m.timestamp, m.value) for m in messages]
        out = engine.process(records)

        # The order matters: events, windows, runs, and only then the commit (a crash in between is repaired by writing again).
        # If MongoDB cannot be reached, a call raises an error: the handler in main() logs it and stops WITHOUT committing.
        started = time.time()
        sink.write_events(out.events)
        sink.write_windows(out.windows)
        sink.write_runs(out.runs)
        stored_at = time.time()

        commit(consumer, topic, engine.commit_offsets(records))

        counts = out.counts
        BATCHES.inc()
        INSERT_SECONDS.observe(stored_at - started)
        MESSAGES.inc(counts["lines"] + counts["smoke"])
        ALERTS.inc(counts["alerts"])
        WINDOWS.inc(counts["windows"])
        UNSEEN.inc(counts["unseen"])
        LATE.inc(counts["late"])
        REORDERED.inc(counts["reordered"])
        REDELIVERED.inc(counts["redelivered"])
        BAD.inc(counts["bad"])

        FLAGGED.labels("logbert").inc(counts["flagged_logbert"])

        for seconds in out.seconds:
            WINDOW_SECONDS.observe(seconds)

        for record in records:
            if record.timestamp_ms and record.timestamp_ms > 0:
                END_TO_END.observe(max(0.0, stored_at - record.timestamp_ms / 1000.0))

        OPEN_LINES.set(len(engine.windower.items) if engine.windower else 0)

        if out.events:
            LAST_MESSAGE.set(stored_at)

        for level, message, fields in out.logs:
            log(level, message, **fields)

        # Only the first ten alert lines are logged one by one; after that the progress line carries the count.
        for event in out.events:
            if event["alert"] and not event.get("smoke") and alerts_total < 10:
                alerts_total += 1
                log("warning", "alert line stored", run=event["run"], offset=event["offset"], truth=event["truth"], template_id=event["template_id"])

        count += counts["lines"] + counts["smoke"]
        windows_total += counts["windows"]

        if count - logged >= 5000:
            log("info", "progress", lines=count, windows=windows_total, topic=topic)
            logged = count

    state.looping = False
    return count


# "Not there yet", in the answer of Kafka: the topic has no partition in it.
class NotThereYet(Exception):
    pass


# Runs action() until it works, and returns what it returned. A failure of a kind in retry_on means "not ready yet": it is logged (the first
# time, then every fifth time, so there is no line every 2 seconds) and action() is tried again after a pause of 2 seconds, then 4, 8 and at
# most 15. A failure of a kind in give_up_on is a mistake in the settings: it ends the program at once, because waiting cannot mend it.
# After `deadline` seconds the program ends too, and Kubernetes starts it again. stop() is true after SIGTERM: the program leaves quietly.
def wait_until(what, action, retry_on, deadline, give_up_on=(), stop=lambda: False):
    started = time.monotonic()
    pause = 2.0
    tries = 0

    while True:
        tries += 1

        try:
            result = action()
        except retry_on as error:
            waited = time.monotonic() - started
            facts = {"tries": tries, "waited_seconds": round(waited), "error": error.__class__.__name__, "detail": str(error)[:300]}

            if isinstance(error, give_up_on):
                log("error", f"cannot go on without {what}: this is a mistake in the settings, and waiting does not mend it", **facts)
                sys.exit(1)

            if waited >= deadline:
                log("error", f"gave up waiting for {what}: the pod stops and Kubernetes starts it again", **facts)
                sys.exit(1)

            if stop():
                log("info", f"stopped while waiting for {what}")
                sys.exit(0)

            if tries == 1 or tries % 5 == 0:
                log("warning", f"waiting for {what}", **facts)

            time.sleep(pause)
            pause = min(pause * 2, 15.0)
            continue

        if tries > 1:
            log("info", f"{what} is ready", tries=tries, waited_seconds=round(time.monotonic() - started))

        return result


# Makes sure that the topic exists, and says whether THIS call made it. Kafka makes a topic when the first message is sent, and the
# processor starts before any producer: version 1.0.0 took the missing topic for an error and the pod stopped, again and again, until
# someone made the topic by hand. "Create it" is safe to repeat: a topic that exists already is left as it is (its number of partitions
# is checked afterwards, in main()). The processor connects with its own certificate; the stack has no Kafka ACLs (README, known
# limitations), so it may create a topic.
def ensure_topic(settings):
    from kafka.admin import KafkaAdminClient
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(
        bootstrap_servers=settings.broker,
        security_protocol="SSL",
        ssl_cafile=settings.ssl_ca,
        ssl_certfile=settings.ssl_cert,
        ssl_keyfile=settings.ssl_key,
        ssl_check_hostname=True,
        client_id="processor-start",
    )

    try:
        admin.create_topics({settings.topic: {"num_partitions": 1, "replication_factor": 1}}, timeout_ms=15000)
    except TopicAlreadyExistsError:
        return False
    finally:
        admin.close()

    return True


# Kafka over TLS with a client certificate. Offsets are committed by hand (see serve).
def connect_consumer(settings):
    from kafka import KafkaConsumer

    return KafkaConsumer(
        settings.topic,
        bootstrap_servers=settings.broker,
        security_protocol="SSL",
        ssl_cafile=settings.ssl_ca,
        ssl_certfile=settings.ssl_cert,
        ssl_keyfile=settings.ssl_key,
        ssl_check_hostname=True,
        value_deserializer=read_json,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=settings.group,

        # A processor that dies without saying goodbye is replaced after this time. (The default is 45 seconds.)
        session_timeout_ms=10000,

        # The longest time between two polls. A batch that completes many windows of a very busy part of the log can take a while
        # (a window of 100,000 lines takes about 10 seconds); the default of 5 minutes is cut generously.
        max_poll_interval_ms=900000,
    )


# The partitions of the topic, once Kafka lists any. (The Kafka library answers with an EMPTY set for a topic that is not there yet.)
def topic_partitions(consumer, topic):
    partitions = consumer.partitions_for_topic(topic)

    if not partitions:
        raise NotThereYet(f"Kafka lists no partition of the topic {topic} yet")

    return partitions


def main():
    settings = Settings()
    start_health_server(settings.metrics_port)

    running = True

    # Kubernetes sends SIGTERM to stop a pod: finish the batch and leave the loop.
    def stop_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def stop():
        return not running

    # Imported here on purpose, so a missing setting is always the first thing reported (and the tests can import this file quickly).
    import torch
    from kafka.errors import KafkaConfigurationError, KafkaError
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    from engine import BundleError, Detectors, Engine
    from sink import MongoSink

    torch.set_num_threads(settings.torch_threads)

    try:
        started = time.time()
        detectors = Detectors.load(settings.bundle)
    except BundleError as error:
        log("error", "the models cannot be used", detail=str(error)[:500])
        sys.exit(1)

    log("info", "models loaded", seconds=round(time.time() - started, 1), logbert_keys=len(detectors.logbert.vocab) - 4, templates=len(detectors.extractor.templates),
        window_seconds=detectors.size_seconds, step_seconds=detectors.step_seconds, g=detectors.g, r=detectors.r, torch_threads=settings.torch_threads)

    # MongoDB over TLS with the user and password. pymongo checks the server certificate against the CA file and the host name.
    client = MongoClient(settings.mongo_host, settings.mongo_port, username=settings.mongo_user, password=settings.mongo_password, tls=True,
                         tlsCAFile=settings.mongo_tls_ca)
    database = client[settings.database]

    # Ask MongoDB to answer once, and make the indexes that the REST API's questions need (creating an index that exists already does
    # nothing). MongoDB may still be starting, and the user may not exist yet: the processor waits for it. Credentials that stay wrong
    # end the wait after START_WAIT_SECONDS, and the log shows the error each time.
    def prepare_mongodb():
        client.admin.command("ping")
        database[settings.events].create_index([("run", 1), ("offset", 1)])
        database[settings.events].create_index("trace_id", sparse=True)
        database[settings.windows].create_index([("run", 1), ("start", 1)])

    wait_until("MongoDB", prepare_mongodb, PyMongoError, settings.start_wait_seconds, stop=stop)

    sink = MongoSink(database, settings.events, settings.windows, settings.runs)

    # The topic first. Kafka makes a topic only when the first message is sent, and this pod starts before any producer; and Kafka itself
    # may not be ready yet. So: wait for Kafka, and make the topic (with ONE partition) if it is not there.
    kafka_errors = (KafkaError, OSError)
    made = wait_until("Kafka, to make sure that the topic exists", lambda: ensure_topic(settings), kafka_errors, settings.start_wait_seconds,
                      KafkaConfigurationError, stop)

    if made:
        log("info", "the topic did not exist yet: the processor made it, with one partition", topic=settings.topic)

    consumer = wait_until("Kafka", lambda: connect_consumer(settings), kafka_errors, settings.start_wait_seconds, KafkaConfigurationError, stop)
    partitions = wait_until("the partitions of the topic", lambda: topic_partitions(consumer, settings.topic), (NotThereYet, *kafka_errors),
                            settings.start_wait_seconds, KafkaConfigurationError, stop)

    # A topic that exists with more than one partition stops the processor for good: the windows need the lines in order.
    if len(partitions) != 1:
        log("error", "the topic must have exactly one partition", topic=settings.topic, partitions=len(partitions))
        sys.exit(1)

    log("info", "processor listening", topic=settings.topic, consumer_group=settings.group, kafka=settings.broker, mongo=f"{settings.mongo_host}:{settings.mongo_port}",
        database=settings.database, collections=[settings.events, settings.windows, settings.runs])

    engine = Engine(detectors, sink.load_state)
    set_ready(True)
    count = 0

    try:
        count = serve(consumer, engine, sink, settings.topic, lambda: not running)
    except Exception as error:
        set_ready(False)
        log("error", "the batch was not stored: the processor stops and Kubernetes restarts it; Kafka hands the batch out again",
            error=error.__class__.__name__, detail=str(error)[:300], topic=settings.topic)

        # No commit for the batch that failed: it must be read again.
        try:
            consumer.close(autocommit=False)
        except Exception:
            pass

        sys.exit(1)

    set_ready(False)
    consumer.close(autocommit=False)
    client.close()
    log("info", "processor stopped", lines=count)


if __name__ == "__main__":
    main()
