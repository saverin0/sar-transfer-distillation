"""teachers.warm_weights - the weight files of the teachers read once ahead from the Hugging Face cache.

A fake cache folder with made-up files of a few kB stands for HF_HOME/hub. No network, no real
weights, nothing fetched. Checks the file lists of the modules against the pinned values their
builds use, the lookup at the pinned revision only, the read of every byte in blocks, one thread
per file at most WARM_WORKERS at once, the largest first, a missing file reported and not fetched,
a file two teachers share read once, a read error reported and not raised, the default cache of
huggingface_hub, and the printed lines (no colon).
"""
import _setup

import inspect
import io
import re
import shutil
import socket
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

import huggingface_hub
from huggingface_hub import constants as hub_constants

from sartransfer.distil import teachers as TEACH
from sartransfer.distil.teachers import dinov3_sat, pe_spatial, siglip2, terrafm

TMP = _setup.TMP_BASE / "warm_weights"
for _ in range(40):                    # Windows may take a moment to let go of a folder
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.25)
TMP.mkdir(parents=True)


def fails(fn, exc, text=""):
    """`fn` must raise `exc` with `text` in its message."""
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in the message {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


# ---- the thread count and its reason, next to the number --------------------------------------
assert TEACH.WARM_WORKERS == 5 and TEACH.WARM_BLOCK == 64 << 20
params = inspect.signature(TEACH.warm_weights).parameters
assert params["workers"].default == TEACH.WARM_WORKERS and params["cache_dir"].default is None
source = Path(TEACH.__file__).read_text(encoding="utf-8")
reason = source.split("WARM_WORKERS = 5")[0].split("# Threads of `warm_weights`")[1]
for words in ("bound per stream", "latency", "23 to 65 MB/s", "279 s", "one big file each", "Not measured"):
    assert words in reason, f"the reason for {TEACH.WARM_WORKERS} threads does not say {words!r}"
assert ":" not in TEACH.warm_weights.__doc__.splitlines()[0]
print("constants: ok")

# ---- the file lists of the modules are the pinned files their builds read ------------------------
FIX, NAF = siglip2.SigLIP2FixRes, siglip2.SigLIP2NaFlex
assert siglip2.FILES == ("config.json", "model.safetensors", "preprocessor_config.json")
assert TEACH.weight_files("siglip2-so400m-256") == tuple((FIX.hf_id, FIX.hf_revision, f) for f in siglip2.FILES)
assert TEACH.weight_files("siglip2-so400m-naflex") == tuple((NAF.hf_id, NAF.hf_revision, f) for f in siglip2.FILES)
assert dinov3_sat.FILES == ("config.json", "model.safetensors")
assert TEACH.weight_files("dinov3-l-sat") == tuple((dinov3_sat.HF_ID, dinov3_sat.HF_REVISION, f) for f in dinov3_sat.FILES)
assert TEACH.weight_files("pe-spatial-l") == ((pe_spatial.HF_ID, pe_spatial.HF_REVISION, "PE-Spatial-L14-448.pt"),)
assert TEACH.weight_files("terrafm-b") == TEACH.weight_files("terrafm-b-radar") == (
    (terrafm.HF_REPO, terrafm.HF_REVISION, "TerraFM-B.pth"),)
for name in TEACH.TEACHERS:                             # every revision pinned to a full commit
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for _, rev, _ in TEACH.weight_files(name)), name
# the builds ask the Hub cache for the same lists (one list per module, nothing typed twice)
assert "allow_patterns=list(FILES)" in inspect.getsource(siglip2.build)
assert "allow_patterns=list(FILES)" in inspect.getsource(dinov3_sat.build)
assert "hf_hub_download(HF_ID, WEIGHTS_FILE, revision=HF_REVISION" in inspect.getsource(pe_spatial.build)
assert "hf_hub_download(HF_REPO, WEIGHTS_FILE, revision=HF_REVISION" in inspect.getsource(terrafm.build)
fails(lambda: TEACH.weight_files("nobody"), KeyError, "unknown teacher")
fails(lambda: siglip2.weight_files(variant="other"), ValueError, "unknown variant")
fails(lambda: terrafm.weight_files(variant="other"), ValueError, "unknown variant")
print("file lists: ok")

# ---- a fake cache, made-up bytes at the pinned revisions. PE's file is only under another revision -
CACHE = TMP / "hub"


def put(repo: str, revision: str, file: str, size: int) -> Path:
    path = CACHE / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((bytes(range(256)) * (size // 256 + 1))[:size])
    return path


SIZES = {(FIX.hf_id, "model.safetensors"): 300_000, (NAF.hf_id, "model.safetensors"): 280_000,
         (dinov3_sat.HF_ID, "model.safetensors"): 200_000, (terrafm.HF_REPO, "TerraFM-B.pth"): 150_000}
NAMES = ["siglip2-so400m-256", "siglip2-so400m-naflex", "dinov3-l-sat", "pe-spatial-l", "terrafm-b-radar", "terrafm-b"]
PATHS = {}
for name in NAMES:
    for repo, revision, file in TEACH.weight_files(name):
        if name == "pe-spatial-l":
            put(repo, "0" * 40, file, 5_000)                     # another revision, so not the pinned file
            continue
        PATHS[(repo, file)] = put(repo, revision, file, SIZES.get((repo, file), 300 + len(file)))
N_FOUND = len(PATHS)                                             # 3 + 3 + 2 + 1, TerraFM once
assert N_FOUND == 9

# ---- no network and nothing fetched while it runs; transformers is not needed ----------------------
NETWORK = []
REAL = {"connect": socket.socket.connect, "lookup": socket.getaddrinfo, "download": huggingface_hub.hf_hub_download,
        "snapshot": huggingface_hub.snapshot_download, "read": TEACH._read_through}


def no_connect(self, address):
    NETWORK.append(str(address))
    raise ConnectionRefusedError("the test must not use the network")


def no_lookup(host, *args, **kwargs):
    NETWORK.append(str(host))
    raise socket.gaierror("the test must not use the network")


def no_fetch(*args, **kwargs):
    raise SystemExit(f"warm_weights fetched something, {args}")


READS = []                                                      # (thread, file name, block)


def slow_read(path, block=TEACH.WARM_BLOCK):
    READS.append((threading.get_ident(), Path(path).name, block))
    time.sleep(0.2)                                             # so the threads overlap
    return REAL["read"](path, block)


had_transformers = "transformers" in sys.modules
socket.socket.connect, socket.getaddrinfo = no_connect, no_lookup
huggingface_hub.hf_hub_download = huggingface_hub.snapshot_download = no_fetch
TEACH._read_through = slow_read
try:
    clock = time.perf_counter()
    with redirect_stdout(io.StringIO()) as buf:
        out = TEACH.warm_weights(NAMES, cache_dir=CACHE, block=4096)
    wall = time.perf_counter() - clock
    printed = buf.getvalue()
    # every found file read once through, all its bytes, the largest first, one thread per file, 5 at once
    assert out["workers"] == 5 and len(out["files"]) == N_FOUND and out["failed"] == []
    assert [e["bytes"] for e in out["files"]] == sorted((p.stat().st_size for p in PATHS.values()), reverse=True)
    assert out["files"][0]["file"] == "model.safetensors" and out["files"][0]["teachers"] == ["siglip2-so400m-256"]
    for e in out["files"]:
        path = PATHS[(e["repo"], e["file"])]
        assert Path(e["path"]) == path and e["bytes"] == path.stat().st_size and e["seconds"] >= 0.2
        assert abs(e["mb_s"] - e["bytes"] / 1e6 / e["seconds"]) < 1e-9
    assert out["bytes"] == sum(p.stat().st_size for p in PATHS.values())
    assert abs(out["mb_s"] - out["bytes"] / 1e6 / out["seconds"]) < 1e-9 and out["seconds"] <= wall
    assert sorted(name for _, name, _ in READS) == sorted(p.name for p in PATHS.values())
    assert all(block == 4096 for *_, block in READS)
    assert 2 <= len({thread for thread, _, _ in READS}) <= 5 and threading.get_ident() not in {t for t, _, _ in READS}
    assert wall < 0.75 * N_FOUND * 0.2, f"{N_FOUND} reads of 0.2 s took {wall:.2f} s, not side by side"
    # the file TerraFM-B shares between its two teachers is read once and names both
    shared = [e for e in out["files"] if e["file"] == "TerraFM-B.pth"]
    assert len(shared) == 1 and shared[0]["teachers"] == ["terrafm-b-radar", "terrafm-b"]
    # PE's file is not in the cache at its pinned revision - reported, not fetched
    assert out["missing"] == [{"teachers": ["pe-spatial-l"], "repo": pe_spatial.HF_ID, "revision": pe_spatial.HF_REVISION,
                               "file": "PE-Spatial-L14-448.pt"}]
    assert f"not in the cache at revision {pe_spatial.HF_REVISION[:8]}, its build fetches it" in printed
    lines = printed.splitlines()
    assert len(lines) == N_FOUND + 1 and all(line.startswith("warm  ") and ":" not in line for line in lines)
    assert sum("MB per second" in line for line in lines) == N_FOUND
    assert re.search(r"^warm  siglip2-so400m-256 +model\.safetensors +0\.3 MB in +\d+\.\d s, \d+ MB per second$",
                     printed, re.M), printed

    # one thread when asked for one, and a file that cannot be read is reported, the others are read
    READS.clear()

    def read_fails_once(path, block=TEACH.WARM_BLOCK):
        if Path(path).name == "TerraFM-B.pth":
            raise OSError("stand-in, the mount dropped the read")
        return slow_read(path, block)

    TEACH._read_through = read_fails_once
    with redirect_stdout(io.StringIO()) as buf:
        out1 = TEACH.warm_weights(["terrafm-b", "dinov3-l-sat"], workers=1, cache_dir=CACHE)
    assert out1["workers"] == 1 and len({thread for thread, _, _ in READS}) == 1
    assert [e["file"] for e in out1["files"]] == ["model.safetensors", "config.json"]
    assert [e["file"] for e in out1["failed"]] == ["TerraFM-B.pth"] and "stand-in" in out1["failed"][0]["error"]
    assert out1["bytes"] == sum(e["bytes"] for e in out1["files"]) and out1["missing"] == []
    assert "TerraFM-B.pth" in buf.getvalue() and "NOT read ahead (OSError, stand-in" in buf.getvalue()
    assert "its build reads it itself" in buf.getvalue()
    TEACH._read_through = slow_read

    # the default cache is the one of huggingface_hub (HF_HOME/hub), here pointed at the fake one
    old_cache = hub_constants.HF_HUB_CACHE
    hub_constants.HF_HUB_CACHE = str(CACHE)
    try:
        with redirect_stdout(io.StringIO()):
            default = TEACH.warm_weights(["dinov3-l-sat"])
    finally:
        hub_constants.HF_HUB_CACHE = old_cache
    assert [e["path"] for e in default["files"]] == [str(PATHS[(dinov3_sat.HF_ID, f)]) for f in ("model.safetensors", "config.json")]
    # nothing asked, nothing read; an empty cache, everything missing
    with redirect_stdout(io.StringIO()):
        none = TEACH.warm_weights([], cache_dir=CACHE)
        empty = TEACH.warm_weights(["pe-spatial-l", "terrafm-b"], cache_dir=TMP / "empty")
    assert none["files"] == [] and none["workers"] == 0 and none["bytes"] == 0
    assert empty["files"] == [] and len(empty["missing"]) == 2 and empty["workers"] == 0
    fails(lambda: TEACH.warm_weights(["nobody"], cache_dir=CACHE), KeyError, "unknown teacher")
finally:
    socket.socket.connect, socket.getaddrinfo = REAL["connect"], REAL["lookup"]
    huggingface_hub.hf_hub_download, huggingface_hub.snapshot_download = REAL["download"], REAL["snapshot"]
    TEACH._read_through = REAL["read"]
assert not NETWORK, f"warm_weights asked for the network, {NETWORK}"
assert ("transformers" in sys.modules) == had_transformers, "warm_weights imported transformers"
print("warm_weights: ok")

# ---- the plain read, block by block, every byte -------------------------------------------------
odd = put("made-up/odd", "1" * 40, "odd.bin", 10_001)
assert TEACH._read_through(odd, block=4096) == 10_001 == TEACH._read_through(odd)
assert TEACH._read_through(put("made-up/odd", "1" * 40, "empty.bin", 0)) == 0
print("_read_through: ok")

shutil.rmtree(TMP, ignore_errors=True)
print("test_warm_weights: all checks passed")
