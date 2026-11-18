# The start-up of the processor (main() in processor.py) without a cluster. Stand-ins take the place of Kafka, MongoDB, the models and the loop;
# everything else in main() is the real code, the waiting included.
#
# Why this file exists: the first run on a real cluster stopped the processor at its check of the topic's partitions. A topic that does not
# exist yet (Kafka makes it when the first message is sent) is answered by the Kafka library with an EMPTY set, and main() took the empty set
# for an error, so the pod crashed again and again (CrashLoopBackOff) until someone made the topic by hand. The tests of the loop
# (test_processor.py) never go through main(), so no test could see it. Version 1.0.1 waits for Kafka, MongoDB and the topic, and makes the
# topic itself; these tests are about that, and they fail on 1.0.0.

import json
import signal
from types import SimpleNamespace

import pytest
from kafka.admin import KafkaAdminClient
from kafka.errors import KafkaConfigurationError, KafkaTimeoutError, TopicAlreadyExistsError
from pymongo.errors import ServerSelectionTimeoutError

import engine
import processor
import sink as sink_module
from fakes import FakeDetectors
from sink import MemorySink

# The settings of the pod, as the ConfigMaps and Secrets give them to it (the metrics port 0: the system picks a free one).
SETTINGS = {"KAFKA_BROKER": "kafka:9092", "KAFKA_TOPIC": "system-logs", "CONSUMER_GROUP": "g", "KAFKA_SSL_CA_FILE": "/ca", "KAFKA_SSL_CERT_FILE": "/c",
            "KAFKA_SSL_KEY_FILE": "/k", "MONGO_TLS_CA_FILE": "/mca", "MONGO_HOST": "mongo", "MONGO_PORT": "27017", "MONGO_DB": "nlp",
            "MONGO_COLLECTION": "events", "MONGO_WINDOWS_COLLECTION": "windows", "MONGO_RUNS_COLLECTION": "runs", "METRICS_PORT": "0",
            "BUNDLE_DIR": "/bundle"}

MADE = "the topic did not exist yet: the processor made it, with one partition"


# Detectors that need no files, with the few facts main() prints about the real ones.
class LoadedModels(FakeDetectors):
    g, r = 3, 2
    logbert = SimpleNamespace(vocab=list(range(10)))
    extractor = SimpleNamespace(templates=list(range(5)))


# What happened when main() ran: what it printed, what it asked of the stand-ins, and how it ended.
class Run:
    def __init__(self):
        self.exit_code = None  # None: main() came back by itself
        self.logs = []
        self.consumer = None
        self.consumer_tries = 0
        self.mongo = None
        self.mongo_commands = []
        self.indexes = []
        self.mongo_closed = False
        self.bundle = None
        self.served = None
        self.threads = []
        self.admin_tries = 0
        self.admin_config = None
        self.admin_closed = False
        self.created = None
        self.asked = []
        self.sleeps = []

    @property
    def messages(self):
        return [line["msg"] for line in self.logs]

    def lines(self, text):
        return [line for line in self.logs if text in line["msg"]]


@pytest.fixture
def start(monkeypatch, tmp_path, capsys):
    (tmp_path / "user").write_text("processor\n")
    (tmp_path / "password").write_text("secret-value\n")

    for name, value in SETTINGS.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setenv("MONGO_USER_FILE", str(tmp_path / "user"))
    monkeypatch.setenv("MONGO_PASSWORD_FILE", str(tmp_path / "password"))
    monkeypatch.delenv("START_WAIT_SECONDS", raising=False)

    # main() installs its own handlers for SIGTERM and SIGINT and starts the health server: give everything back afterwards.
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    servers = []
    real_start = processor.start_health_server

    def start_health_server(port):
        servers.append(real_start(port))
        return servers[-1]

    monkeypatch.setattr(processor, "start_health_server", start_health_server)

    # partitions: what Kafka says about the topic. One answer (a set, or None), or a list of answers, one for each time it is asked (the last
    # one is repeated). topic_exists: the admin client answers "TopicAlreadyExists". The three *_failures: how many tries fail before one works.
    def run(partitions={0}, models=LoadedModels, serve_error=None, topic_exists=False, admin_failures=0, consumer_failures=0, mongo_failures=0,
            admin_error=KafkaTimeoutError, wait_seconds=None, on_sleep=None):
        result = Run()
        answers = partitions if isinstance(partitions, list) else [partitions]

        if wait_seconds is not None:
            monkeypatch.setenv("START_WAIT_SECONDS", str(wait_seconds))

        class Admin:
            def __init__(self, **config):
                result.admin_tries += 1

                if result.admin_tries <= admin_failures:
                    raise admin_error("Unable to bootstrap from kafka:9092")

                result.admin_config = config

            def create_topics(self, topics, **options):
                result.created = (topics, options)

                if topic_exists:
                    raise TopicAlreadyExistsError("the topic exists already")

            def close(self):
                result.admin_closed = True

        class Consumer:
            def __init__(self, *topics, **config):
                result.consumer_tries += 1

                if result.consumer_tries <= consumer_failures:
                    raise KafkaTimeoutError("Unable to bootstrap from kafka:9092")

                result.consumer = self
                self.topics, self.config, self.closed_with = topics, config, None

            def partitions_for_topic(self, topic):
                result.asked.append(topic)
                return answers[min(len(result.asked), len(answers)) - 1]

            def close(self, autocommit=True):
                self.closed_with = {"autocommit": autocommit}

        class Collection:
            def __init__(self, name):
                self.name = name

            def create_index(self, keys, **options):
                result.indexes.append((self.name, keys))

        class Database:
            def __getitem__(self, name):
                return Collection(name)

        class Client:
            def __init__(self, host, port, **config):
                result.mongo = {"host": host, "port": port, **config}

                def command(name):
                    result.mongo_commands.append(name)

                    if len(result.mongo_commands) <= mongo_failures:
                        raise ServerSelectionTimeoutError("mongo:27017: connection refused")

                self.admin = SimpleNamespace(command=command)

            def __getitem__(self, database):
                return Database()

            def close(self):
                result.mongo_closed = True

        class Detectors:
            @staticmethod
            def load(bundle):
                result.bundle = bundle

                if models is None:
                    raise engine.BundleError("model.pt is missing from the bundle")

                return models()

        def serve(consumer, engine_, sink_, topic, stop, max_records=500):
            result.served = {"topic": topic, "consumer": consumer, "ready": processor.state.ready}

            if serve_error:
                raise serve_error

            return 12

        def sleep(seconds):
            result.sleeps.append(seconds)

            if on_sleep:
                on_sleep()

        monkeypatch.setattr("torch.set_num_threads", result.threads.append)
        monkeypatch.setattr("kafka.KafkaConsumer", Consumer)
        monkeypatch.setattr("kafka.admin.KafkaAdminClient", Admin)
        monkeypatch.setattr("pymongo.MongoClient", Client)
        monkeypatch.setattr(engine, "Detectors", Detectors)
        monkeypatch.setattr(sink_module, "MongoSink", lambda *args: MemorySink())
        monkeypatch.setattr(processor, "serve", serve)
        monkeypatch.setattr(processor.time, "sleep", sleep)

        try:
            processor.main()
        except SystemExit as stop:
            result.exit_code = stop.code

        result.logs = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        return result

    yield run

    for server in servers:
        server.shutdown()
        server.server_close()

    for number, handler in handlers.items():
        signal.signal(number, handler)

    processor.set_ready(False)
    processor.state.looping = False


# ------------------------------------------------------------------------------------------------ the mistake of the first cluster run

# Kafka makes a topic when the first message is sent, so a processor that starts first sees a topic that is not there yet. 1.0.0 stopped.
# 1.0.1 asks Kafka for the topic (one partition), says that it did, and goes on.
def test_a_topic_that_does_not_exist_yet_is_made_by_the_processor_and_the_processor_goes_on(start):
    run = start(partitions={0})

    assert run.exit_code is None
    assert run.served and run.served["topic"] == "system-logs" and run.served["ready"] is True
    assert "processor listening" in run.messages

    # it asked for ONE partition and one copy, as the only way to keep the lines of a window in order
    topics, _ = run.created
    assert topics == {"system-logs": {"num_partitions": 1, "replication_factor": 1}}
    assert run.admin_closed

    (line,) = run.lines(MADE)
    assert line["level"] == "info" and line["topic"] == "system-logs"
    assert run.sleeps == []


# The library says "not there yet" with an EMPTY set (older versions with None), also for a moment after the topic was made.
@pytest.mark.parametrize("not_yet", [set(), None], ids=["empty set", "None"])
def test_a_topic_that_Kafka_does_not_list_yet_is_waited_for(start, not_yet):
    run = start(partitions=[not_yet, not_yet, {0}])

    assert run.exit_code is None and run.served["ready"] is True
    assert run.sleeps == [2.0, 4.0] and len(run.asked) == 3
    (line,) = run.lines("waiting for the partitions of the topic")
    assert line["level"] == "warning" and line["tries"] == 1 and line["error"] == "NotThereYet"
    (line,) = run.lines("the partitions of the topic is ready")
    assert line["tries"] == 3


def test_a_topic_that_exists_with_one_partition_starts_the_processor_without_a_remark(start):
    run = start(partitions={0}, topic_exists=True)

    assert run.exit_code is None and run.served["ready"] is True
    assert "processor listening" in run.messages
    assert MADE not in run.messages and run.admin_closed
    assert not any(line["level"] in ("warning", "error") for line in run.logs)


@pytest.mark.parametrize("partitions", [{0, 1}, {0, 1, 2}])
def test_a_topic_with_more_than_one_partition_stops_the_processor_for_good(start, partitions):
    run = start(partitions=partitions, topic_exists=True)

    assert run.exit_code == 1
    assert run.served is None and processor.state.ready is False
    assert "processor listening" not in run.messages
    assert run.sleeps == []

    (line,) = run.lines("the topic must have exactly one partition")
    assert line["level"] == "error" and line["topic"] == "system-logs" and line["partitions"] == len(partitions)


# ------------------------------------------------------------------------------------------------ waiting for Kafka and MongoDB

def test_kafka_that_does_not_answer_yet_is_waited_for(start):
    run = start(admin_failures=3)

    assert run.exit_code is None and run.served["ready"] is True
    assert run.admin_tries == 4
    assert run.sleeps == [2.0, 4.0, 8.0]

    # one line when the waiting starts (not one for every try), one when it is over
    (line,) = run.lines("waiting for Kafka, to make sure that the topic exists")
    assert line["level"] == "warning" and line["tries"] == 1 and line["error"] == "KafkaTimeoutError"
    (line,) = run.lines("is ready")
    assert line["msg"].startswith("Kafka, to make sure") and line["tries"] == 4


def test_the_pause_grows_to_fifteen_seconds_and_the_log_says_it_every_fifth_try(start):
    run = start(admin_failures=11)

    assert run.sleeps == [2.0, 4.0, 8.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    assert [line["tries"] for line in run.lines("waiting for Kafka, to make sure")] == [1, 5, 10]


def test_a_consumer_that_cannot_connect_yet_is_waited_for(start):
    run = start(consumer_failures=2)

    assert run.exit_code is None and run.served["ready"] is True
    assert run.consumer_tries == 3 and run.sleeps == [2.0, 4.0]
    (line,) = run.lines("waiting for Kafka")
    assert line["tries"] == 1 and line["msg"] == "waiting for Kafka"


def test_mongodb_that_does_not_answer_yet_is_waited_for(start):
    run = start(mongo_failures=2)

    assert run.exit_code is None and run.served["ready"] is True
    assert run.mongo_commands == ["ping", "ping", "ping"] and run.sleeps == [2.0, 4.0]
    (line,) = run.lines("waiting for MongoDB")
    assert line["level"] == "warning" and line["error"] == "ServerSelectionTimeoutError"

    # the indexes were made after the answer, once each
    assert sorted(name for name, _ in run.indexes) == ["events", "events", "windows"]

    # and Kafka was not touched before MongoDB answered
    assert run.admin_tries == 1


def test_the_processor_gives_up_after_the_waiting_time_and_says_so(start):
    run = start(admin_failures=1000, wait_seconds=0)

    assert run.exit_code == 1
    assert run.consumer is None and run.served is None and processor.state.ready is False

    (line,) = run.lines("gave up waiting for Kafka, to make sure that the topic exists")
    assert line["level"] == "error" and line["tries"] == 1 and line["error"] == "KafkaTimeoutError" and "kafka:9092" in line["detail"]


def test_a_wrong_setting_of_Kafka_is_not_waited_for(start):
    run = start(admin_failures=1000, admin_error=KafkaConfigurationError)

    assert run.exit_code == 1 and run.sleeps == []
    (line,) = run.lines("cannot go on without Kafka")
    assert line["level"] == "error" and line["error"] == "KafkaConfigurationError"


def test_a_mistake_in_the_program_is_not_waited_for(start):
    with pytest.raises(ZeroDivisionError):
        start(admin_failures=1000, admin_error=ZeroDivisionError)


# A pod that Kubernetes stops while it waits leaves at once and quietly (SIGTERM), and does not start the loop.
def test_a_stop_signal_while_waiting_ends_the_waiting(start):
    run = start(admin_failures=1000, on_sleep=lambda: signal.raise_signal(signal.SIGTERM))

    assert run.exit_code == 0 and len(run.sleeps) == 1
    assert run.served is None and run.consumer is None
    assert run.lines("stopped while waiting for Kafka, to make sure that the topic exists")


# ------------------------------------------------------------------------------------------------ the rest of main(), unchanged

def test_the_processor_connects_the_way_the_cluster_needs(start):
    run = start({0})

    # Kafka: TLS with the processor's own certificate, the host name checked, and offsets that only the loop commits (after the writes)
    consumer = run.consumer
    assert consumer.topics == ("system-logs",) and run.asked == ["system-logs"]
    assert consumer.config["bootstrap_servers"] == "kafka:9092" and consumer.config["group_id"] == "g"
    assert consumer.config["security_protocol"] == "SSL" and consumer.config["ssl_check_hostname"] is True
    assert (consumer.config["ssl_cafile"], consumer.config["ssl_certfile"], consumer.config["ssl_keyfile"]) == ("/ca", "/c", "/k")
    assert consumer.config["enable_auto_commit"] is False and consumer.config["auto_offset_reset"] == "earliest"

    # the administrator that makes the topic uses the same TLS and the same certificate, and only settings that the library knows
    admin = run.admin_config
    assert admin["bootstrap_servers"] == "kafka:9092" and admin["security_protocol"] == "SSL" and admin["ssl_check_hostname"] is True
    assert (admin["ssl_cafile"], admin["ssl_certfile"], admin["ssl_keyfile"]) == ("/ca", "/c", "/k")
    assert set(admin) <= set(KafkaAdminClient.DEFAULT_CONFIG)

    # MongoDB: TLS with the CA file, its own account, asked once at the start, and the three indexes of the REST API's questions
    assert (run.mongo["host"], run.mongo["port"], run.mongo["tls"], run.mongo["tlsCAFile"]) == ("mongo", 27017, True, "/mca")
    assert (run.mongo["username"], run.mongo["password"]) == ("processor", "secret-value")
    assert run.mongo_commands == ["ping"]
    assert sorted(name for name, _ in run.indexes) == ["events", "events", "windows"]

    assert run.bundle == "/bundle" and run.threads == [2]


def test_the_waiting_time_can_be_set_and_is_twenty_minutes_by_default(monkeypatch):
    for name, value in SETTINGS.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setenv("MONGO_USER", "u")
    monkeypatch.setenv("MONGO_PASSWORD", "p")
    monkeypatch.delenv("MONGO_USER_FILE", raising=False)
    monkeypatch.delenv("MONGO_PASSWORD_FILE", raising=False)
    monkeypatch.delenv("START_WAIT_SECONDS", raising=False)
    assert processor.Settings().start_wait_seconds == 1200

    monkeypatch.setenv("START_WAIT_SECONDS", "90")
    assert processor.Settings().start_wait_seconds == 90


def test_a_normal_end_stores_nothing_more_and_closes_without_committing(start):
    run = start({0})

    assert run.exit_code is None and run.mongo_closed
    assert run.consumer.closed_with == {"autocommit": False}
    assert processor.state.ready is False

    (line,) = run.lines("processor stopped")
    assert line["lines"] == 12


def test_a_failed_batch_stops_the_processor_without_committing(start):
    run = start({0}, serve_error=ConnectionError("MongoDB is gone"))

    assert run.exit_code == 1
    assert run.consumer.closed_with == {"autocommit": False}
    assert processor.state.ready is False
    assert "processor stopped" not in run.messages

    (line,) = [line for line in run.logs if line["msg"].startswith("the batch was not stored")]
    assert line["level"] == "error" and line["error"] == "ConnectionError" and "MongoDB is gone" in line["detail"]


def test_models_that_cannot_be_used_stop_the_processor_before_it_touches_kafka_or_mongodb(start):
    run = start({0}, models=None)

    assert run.exit_code == 1
    assert run.consumer is None and run.mongo is None and run.served is None and run.admin_tries == 0

    (line,) = run.lines("the models cannot be used")
    assert line["level"] == "error" and "model.pt" in line["detail"]


# ------------------------------------------------------------------------------------------------ the waiting helper on its own

def test_wait_until_returns_what_the_action_returns_without_a_word(capsys):
    assert processor.wait_until("a thing", lambda: 42, ValueError, 10) == 42
    assert capsys.readouterr().out == ""


def test_wait_until_does_not_wait_for_an_error_of_another_kind():
    def action():
        raise KeyError("a mistake in the program")

    with pytest.raises(KeyError):
        processor.wait_until("a thing", action, ValueError, 10)
