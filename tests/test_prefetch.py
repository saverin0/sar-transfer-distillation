"""score.Prefetch, score.read_ahead and score.model_files - the downloads of notebook 03 started at the start.

Tiny files in a temp folder on D, jobs that are plain functions held back by events, no network, no torch
model, nothing fetched. Checks the order of the items in a lane and two lanes side by side, a consumer
that waits only for its own item while another is held back, the seconds it waited, first_ready, a failing
job that fails its own item only with the token hidden, what a job prints kept with its item and printed
once by the first wait while the main thread prints through, the streams given back, close, the record,
the summary and the waited line, the errors of a wrong call; read_ahead reads every byte, one thread per
file and at most `workers` at once, a missing file raised after the others; model_files finds the files
of a local folder and of the Hugging Face cache at the revision main. No colon in a printed line.
"""
import _setup

import inspect
import io
import shutil
import sys
import threading
import time
from pathlib import Path

from sartransfer.distil import score as SC

TMP = _setup.TMP_BASE / "prefetch"
for _ in range(40):                    # Windows may take a moment to let go of a folder
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.25)
TMP.mkdir(parents=True)
SECRET = "MADE-UP-TOKEN-OF-THE-PREFETCH-TEST"


def fails(fn, exc, text=""):
    """`fn` must raise `exc` with `text` in its message."""
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in the message {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


def no_colon(text: str) -> None:
    bad = [line for line in text.splitlines() if ":" in line and "\\" not in line]
    assert not bad, f"a printed line holds a colon, {bad[:3]}"


# ---- the thread count and its reason, next to the number --------------------------------------
assert SC.READ_AHEAD_WORKERS == 5 and SC.READ_BLOCK == 64 << 20 and SC.PREFETCH_POLL == 1.0
assert inspect.signature(SC.read_ahead).parameters["workers"].default == SC.READ_AHEAD_WORKERS
source = Path(SC.__file__).read_text(encoding="utf-8")
reason = source.split("READ_AHEAD_WORKERS = 5")[0].split("# Threads of `read_ahead`")[1]
for words in ("bound per stream", "latency", "23 to 65 MB/s", "side by side add up", "Not measured"):
    assert words in reason, f"the reason for {SC.READ_AHEAD_WORKERS} threads does not say {words!r}"
for fn in (SC.read_ahead, SC.model_files, SC.Prefetch.wait, SC.Prefetch.first_ready, SC.Prefetch.close):
    assert ":" not in fn.__doc__.splitlines()[0], fn
print("constants: ok")

# ---- read_ahead - every byte, one thread per file, at most `workers` at once ----------------------
files = []
for k, size in enumerate((5_000, 12_345, 1, 0, 70_000)):
    path = TMP / f"file_{k}.bin"
    path.write_bytes(bytes(range(256)) * (size // 256) + bytes(size % 256))
    files.append(path)
got = SC.read_ahead(files, block=1000)
assert got["bytes"] == sum(p.stat().st_size for p in files) == 87_346 and got["files"] == 5 and got["seconds"] >= 0
assert SC.read_ahead([str(files[0])])["bytes"] == 5_000          # a path as text
LIVE = {"now": 0, "most": 0, "opened": []}
lock = threading.Lock()


def slow_open(path, *args, **kwargs):
    """open, holding the file a moment, counting the files open at once."""
    with lock:
        LIVE["now"] += 1
        LIVE["most"] = max(LIVE["most"], LIVE["now"])
        LIVE["opened"].append(Path(path).name)
    time.sleep(0.15)
    f = open(path, *args, **kwargs)
    real_close = f.close

    def close():
        with lock:
            LIVE["now"] -= 1
        real_close()

    f.close = close
    return f


SC.open = slow_open                    # the module's open, in front of the builtin
try:
    SC.read_ahead(files, workers=2)
    assert LIVE["most"] == 2 and sorted(LIVE["opened"]) == sorted(p.name for p in files), LIVE
    LIVE.update(now=0, most=0, opened=[])
    SC.read_ahead(files[:3])
    assert LIVE["most"] == 3, "one thread per file"
    LIVE.update(now=0, most=0, opened=[])
    missing = TMP / "not_there.bin"
    fails(lambda: SC.read_ahead([files[0], missing, files[1]]), FileNotFoundError, "not_there.bin")
    assert sorted(LIVE["opened"]) == sorted([files[0].name, files[1].name]), "the other files were read"
finally:
    del SC.open
fails(lambda: SC.read_ahead([]), FileNotFoundError, "no file to read ahead")
print("read_ahead: ok")

# ---- model_files - a local folder, the Hugging Face cache at main, nothing found ------------------
folder = TMP / "local_model"
folder.mkdir()
(folder / "config.json").write_text("{}", encoding="utf-8")
(folder / "model.safetensors").write_bytes(b"x" * 100)
assert SC.model_files(str(folder)) == [folder / "config.json", folder / "model.safetensors"]
(folder / "model.safetensors").unlink()
assert SC.model_files(folder) == [folder / "config.json"], "a file that is not there is left out"
cache = TMP / "hub"
sha = "0123456789abcdef0123456789abcdef01234567"
repo = cache / "models--made-up-org--made-up-model"
(repo / "refs").mkdir(parents=True)
(repo / "refs" / "main").write_text(sha, encoding="utf-8")
(repo / "snapshots" / sha).mkdir(parents=True)
for name, size in (("config.json", 10), ("model.safetensors", 300)):
    (repo / "snapshots" / sha / name).write_bytes(b"y" * size)
found = SC.model_files("made-up-org/made-up-model", cache_dir=cache)
assert [p.name for p in found] == ["config.json", "model.safetensors"] and all(p.is_file() for p in found)
assert found[1].parent.name == sha, "the revision main"
assert SC.read_ahead(found)["bytes"] == 310
assert SC.model_files("made-up-org/another-model", cache_dir=cache) == []
assert SC.model_files("made-up-org/made-up-model", names=("other.bin",), cache_dir=cache) == []
print("model_files: ok")

# ---- Prefetch -------------------------------------------------------------------------------------
REAL_OUT, REAL_ERR = sys.stdout, sys.stderr


class Switch:
    """sys.stdout and sys.stderr of this test, so what the main thread prints can be read back."""

    def __init__(self):
        self.text = io.StringIO()

    def write(self, text):
        return self.text.write(text)

    def flush(self):
        pass

    def take(self) -> str:
        out = self.text.getvalue()
        self.text = io.StringIO()
        return out


main = Switch()
gates = {name: threading.Event() for name in ("snow", "aerial", "late")}
RAN = []


def job(name, size=0, gate=None, say=(), err=None, source="Hugging Face"):
    def run():
        RAN.append(name)
        for line in say:
            print(line)
        if err is not None:
            print(f"warning of {name}", file=sys.stderr)
        if gate is not None:
            assert gates[gate].wait(20), f"{gate} was never let go"
        if err is not None:
            raise err
        return {"source": source, "bytes": size}
    return run


sys.stdout = sys.stderr = main
try:
    fetch = SC.Prefetch(secrets=(SECRET, None, ""))
    assert fetch.secrets == (SECRET,)
    fetch.add("forty_v1", job("forty_v1", 520_000_000, say=("forty_v1  from Hugging Face in 0.1 min",)))
    fetch.add("glavitu_alp", job("glavitu_alp", err=FileNotFoundError(f"stand-in, not there, token {SECRET}")))
    fetch.add("snow", job("snow", 7_000_000_000, gate="snow", say=("snow: 2 splits ready",)))
    fetch.add("flair_hub", job("flair_hub", 0, source="on the runtime disk already"))
    fetch.add("weights", job("weights", 1_212_000_000, say=("read 2 files",), source="the model cache on Drive"),
              lane="read-ahead")
    fetch.add("students", job("students", gate="late"), lane="read-ahead")
    fails(lambda: fetch.add("snow", job("snow")), ValueError, "added twice")
    fails(lambda: fetch.wait("snow", by="x"), RuntimeError, "call start() first")
    assert fetch.names == ["forty_v1", "glavitu_alp", "snow", "flair_hub", "weights", "students"]
    assert fetch.lanes() == {"data": ["forty_v1", "glavitu_alp", "snow", "flair_hub"],
                             "read-ahead": ["weights", "students"]}
    assert "snow" in fetch and "other" not in fetch
    assert fetch.start() is fetch
    assert isinstance(sys.stdout, SC._ThreadLines) and sys.stdout.base is main, "the streams are wrapped"
    fails(lambda: fetch.add("late", job("late")), RuntimeError, "before start()")
    fails(fetch.start, RuntimeError, "started already")
    fails(lambda: fetch.wait("other", by="x"), KeyError, "not an item of this fetch")
    fails(lambda: fetch.first_ready([], by="x"), ValueError, "no item to wait for")

    # the control waits for forest and the weights only; snow is held back the whole time
    print("main thread line one")
    forest = fetch.wait("forty_v1", by="control")
    weights = fetch.wait("weights", by="control")
    said = main.take()
    assert forest["ok"] and forest["status"] == "ready" and forest["source"] == "Hugging Face"
    assert forest["bytes"] == 520_000_000 and forest["lane"] == "data" and forest["ready_at"] >= forest["started"]
    assert weights["ok"] and fetch.record("snow")["status"] == "running", "snow is still being fetched"
    assert fetch.record("flair_hub")["status"] == "waiting", "the lane takes one item at a time, in its order"
    assert fetch.record("students")["status"] == "running"
    lines = said.splitlines()
    assert lines[0] == "main thread line one" and lines[1] == "forty_v1  from Hugging Face in 0.1 min", lines
    assert lines[2].startswith("forty_v1  ready after ") and ", from Hugging Face, 520 MB in " in lines[2]
    assert lines[2].endswith(" MB per second") and lines[3].startswith("control waited ")
    assert lines[3].endswith(" s for forty_v1") and lines[4] == "read 2 files"
    assert lines[5].startswith("weights  ready after ") and ", from the model cache on Drive, 1,212 MB in" in lines[5]
    assert "snow: 2 splits ready" not in said, "a line of an item that is not through"
    assert fetch.waited["control"] < 5

    # glavitu_alp failed on its own, its line comes with the first wait; the token is hidden
    alp = fetch.wait("glavitu_alp", by="SAR runs")
    said = main.take()
    assert alp["status"] == "failed" and not alp["ok"] and SECRET not in alp["error"] and "<token>" in alp["error"]
    assert "warning of glavitu_alp" in said, "what the job wrote to stderr is kept with its item"
    assert "glavitu_alp  NOT AVAILABLE after " in said and "FileNotFoundError, stand-in, not there, token <token>" in said
    assert SECRET not in said
    assert fetch.record("snow")["status"] == "running", "the failure did not stop the lane"

    # first_ready returns at once with what is through, snow still held back
    ready = fetch.first_ready(["snow", "forty_v1", "glavitu_alp"], by="SAR runs")
    assert ready == ["forty_v1", "glavitu_alp"], ready
    said = main.take()
    assert said.startswith("SAR runs waited 0.0 s, on the runtime now forty_v1, glavitu_alp"), said
    assert "ready after" not in said, "the lines of an item are printed once"

    # a consumer that waits for snow blocks until it is let go, and the seconds are counted
    threading.Timer(0.6, gates["snow"].set).start()
    ready = fetch.first_ready(["snow"], by="SAR runs")
    said = main.take()
    assert ready == ["snow"] and fetch.waited["SAR runs"] >= 0.5, fetch.waited
    assert said.splitlines()[0] == "snow: 2 splits ready" and "snow  ready after " in said
    assert ", 7,000 MB in " in said
    rec = fetch.wait("flair_hub", by="aerial runs")
    said = main.take()
    assert rec["ok"] and rec["bytes"] == 0 and rec["mb_s"] is None
    assert "flair_hub  ready after " in said and said.splitlines()[0].endswith(", on the runtime disk already")
    assert RAN.index("forty_v1") < RAN.index("glavitu_alp") < RAN.index("snow") < RAN.index("flair_hub")
    data = [fetch.record(n)["ready_at"] for n in ("forty_v1", "glavitu_alp", "snow", "flair_hub")]
    assert data == sorted(data), "the data lane is ready in the order its items were added"

    # the read-ahead lane is still at its last item, so the streams stay wrapped until it ends
    assert isinstance(sys.stdout, SC._ThreadLines)
    threading.Timer(0.3, gates["late"].set).start()
    fetch.close()
    assert sys.stdout is main and sys.stderr is main, "the streams of before are back"
    fetch.close()                                          # a second close does nothing more
    stud = fetch.wait("students", by="aerial runs")
    assert stud["ok"] and stud["source"] == "Hugging Face"
    main.take()

    table = fetch.summary()
    assert list(table["item"]) == fetch.names and set(table["status"]) == {"ready", "failed"}
    assert list(table.columns) == ["item", "lane", "status", "source", "MB", "seconds", "MB_per_s", "started_s",
                                   "ready_s", "error"]
    assert abs(table.loc[table["item"] == "snow", "MB"].iloc[0] - 7000) < 1e-9
    line = fetch.waited_line()
    total = sum(fetch.waited.values())
    own = sum(fetch.record(n)["seconds"] for n in fetch.names)
    assert line.startswith(f"the GPU work waited {total:.1f} s for data (control ") and "SAR runs" in line
    assert f"the 6 items take {own:.1f} s (their own times added up, timed side by side), {own - total:.1f} s saved" \
        in line and ":" not in line
    print(table.to_string(), file=REAL_OUT)
    print(line, file=REAL_OUT)

    # close stops the items that have not started, the running one ends first
    gates["aerial"].clear()
    stop = SC.Prefetch()
    stop.add("held", job("held", 10, gate="aerial"))
    stop.add("next", job("next", 10))
    stop.add("other", job("other", 10), lane="read-ahead")
    stop.start()
    stop.wait("other", by="x")
    threading.Timer(0.3, gates["aerial"].set).start()
    stop.close()
    assert stop.record("held")["ok"] and stop.record("next")["status"] == "stopped"
    assert "next" not in RAN and sys.stdout is main
    stop.wait("next", by="x")
    assert "next  NOT AVAILABLE, the fetch was stopped before this item started" in main.take()

    # a fetch without items gives the streams back at once; a job that returns no dict is through with 0 bytes
    empty = SC.Prefetch().start()
    assert sys.stdout is main and empty.waited_line().startswith("the GPU work waited 0.0 s for data (no consumer)")
    odd = SC.Prefetch()
    odd.add("none", lambda: None)
    odd.start()
    assert odd.wait("none", by="x")["ok"] and odd.record("none")["bytes"] == 0
    odd.close()
    assert main.take().splitlines() == ["none  ready after " + f"{odd.record('none')['ready_at']:.1f} s, nothing fetched",
                                        "x waited " + f"{odd.waited['x']:.1f} s for none"]

    # the main thread prints while a lane prints, every line lands where it belongs
    talk = SC.Prefetch()

    def chatty():
        for i in range(200):
            print(f"lane line {i}")
            if i % 50 == 0:
                time.sleep(0.01)
        return {"source": "Drive", "bytes": 1}

    talk.add("chatty", chatty)
    talk.start()
    for i in range(200):
        print(f"main line {i}")
    main_said = main.take()
    talk.wait("chatty", by="x")
    talk.close()
    item_said = main.take()
    assert main_said.splitlines() == [f"main line {i}" for i in range(200)], "a lane line reached the main thread"
    assert item_said.splitlines()[:200] == [f"lane line {i}" for i in range(200)]
    no_colon(said + item_said)
finally:
    sys.stdout, sys.stderr = REAL_OUT, REAL_ERR
assert not [t for t in threading.enumerate() if t.name.startswith("prefetch ")], "a lane thread is left"
print("Prefetch: ok")

shutil.rmtree(TMP, ignore_errors=True)
print("test_prefetch: all checks passed")
