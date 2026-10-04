"""Shared start of every test here: sartransfer from THIS repo's src, temp files on D:.

Run a test as a script, from any folder:   python tests/test_records.py

- sys.path gets <repo>/src first, and the import is checked: a sartransfer from anywhere
  else (an installed copy, another repository) stops the test. The path is printed.
- Temp files go under <system temp>/sartransfer-distillation-tests. The system temp folder comes
  from TMPDIR / TEMP / TMP and must not be on C: -- set them to a folder on D: first.
- Optional, only for the tests that use them:
    SARTRANSFER_TEST_PYLIB    folder(s) added at the END of sys.path (e.g. one holding timm
                              and einops when they are not installed)
    SARTRANSFER_TEST_WEIGHTS  folder(s) with real model weights; tests that compare against
                              real weights are skipped without it
    SARTRANSFER_TEST_NET      set to 1 to allow the tests that read a few real Major-TOM rows
                              over the network; they are skipped without it
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
for _extra in filter(None, os.environ.get("SARTRANSFER_TEST_PYLIB", "").split(os.pathsep)):
    if _extra not in sys.path:
        sys.path.append(_extra)

import sartransfer  # noqa: E402

ORIGIN = Path(sartransfer.__file__).resolve()
if SRC.resolve() not in ORIGIN.parents:
    raise SystemExit(f"IMPORT GUARD: sartransfer imported from {ORIGIN}, not from {SRC}")
print(f"import guard ok: sartransfer {sartransfer.__version__} from {ORIGIN.parent}")

# Checked BEFORE tempfile.gettempdir(), which probes its folder by writing a small file there.
_first = next((os.environ[k] for k in ("TMPDIR", "TEMP", "TMP") if os.environ.get(k)), "")
if not _first or not Path(_first).is_dir() or Path(_first).resolve().drive.upper() == "C:":
    raise SystemExit(f"temp folder {_first or '(TMPDIR/TEMP/TMP not set)'} is missing or on C:; "
                     "set TMPDIR, TEMP and TMP to an existing folder on D: first")
TMP_ROOT = Path(tempfile.gettempdir()).resolve()
if TMP_ROOT.drive.upper() == "C:":
    raise SystemExit(f"temp folder {TMP_ROOT} is on C:; set TMPDIR, TEMP and TMP to a folder on D: first")
TMP_BASE = TMP_ROOT / "sartransfer-distillation-tests"
TMP_BASE.mkdir(parents=True, exist_ok=True)
WEIGHTS = [Path(p) for p in os.environ.get("SARTRANSFER_TEST_WEIGHTS", "").split(os.pathsep) if p]
NET = os.environ.get("SARTRANSFER_TEST_NET", "") == "1"

# The last cell of every notebook, word for word - flush Drive, print, release the machine.
FLUSH_SOURCE = '# Flush Drive and release the runtime. The run is over when this cell prints, Drive then holds\n# everything that was written. With END_RUNTIME the Colab machine is released right after, so\n# the GPU stops and no compute is spent waiting - google.colab.runtime.unassign asks Colab from\n# the machine itself (an HTTP request to the runtime service, read in the colabtools source on\n# 2026-10-04), so it works from VS Code as well as from the browser. A cell that fails earlier\n# stops the run before this one, so a runtime with an error stays up to be looked at. Set\n# END_RUNTIME = False to keep the runtime; run the mount cell again to use Drive afterwards.\nfrom google.colab import drive, runtime\n\nEND_RUNTIME = True               # release the machine after the flush, the outputs stay in the notebook\n\ndrive.flush_and_unmount()\nprint("Drive flushed and unmounted, the run is over"\n      + (", the runtime is released now" if END_RUNTIME else ", the runtime stays up"), flush=True)\nif END_RUNTIME:\n    runtime.unassign()\n'

def check_sync_cell(nb: dict, SYNC) -> str:
    """The sync cell of a notebook holds the current src, packed, or is cleared (a published copy carries no
    packed code). Returns "packed" or "cleared"; a packed cell that is not current stops the test."""
    import base64
    import gzip
    import io
    import re
    import tarfile

    i = SYNC.cell_index(nb)
    assert i is not None, "the notebook has no sync cell"
    lines = "".join(nb["cells"][i]["source"]).splitlines()
    assert lines[0] == SYNC.BEGIN and lines[-1] == SYNC.END
    if SYNC.is_cleared(nb):
        return "cleared"
    packed, now = SYNC.embedded_fingerprint(nb), SYNC.source_fingerprint(SYNC.collect_sources())
    assert packed == now, (f"the sync cell holds the source {packed}, src is {now}. "
                           "Run   python sync.py   and then this test again")
    payload = "".join(re.findall(r'^    "([A-Za-z0-9+/=]+)"$', "".join(nb["cells"][i]["source"]), re.M))
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(payload)))) as packed_tar:
        unpacked = {m.name: packed_tar.extractfile(m).read() for m in packed_tar.getmembers()}
    assert unpacked == SYNC.collect_sources(), "the packed source differs from src"
    return "packed"

def _private(name: str) -> tuple:
    """A tuple of tests/_private_names.py, a private-only file; empty where it is not (a public copy)."""
    try:
        import _private_names
    except ImportError:
        return ()
    return tuple(getattr(_private_names, name))


def private_names() -> tuple:
    """Names that may not appear in a public file - other repositories, accounts and folders of the author."""
    return _private("NAMES")


def private_only() -> tuple:
    """The private-only files of the repository (left out of a public copy) and how a file could point to them."""
    return _private("PRIVATE_ONLY"), _private("POINTERS")


def is_public_copy() -> bool:
    """True in a public copy, which carries no file of private names."""
    return not (REPO / "tests" / "_private_names.py").is_file()
