"""Pack the package source into a marked notebook cell, so Colab needs no git.

The Colab runtime cannot see local files. Cloning from GitHub works but forces a
push for every change to `src/`. This module removes that: the source travels
*inside* the `.ipynb`, which the Colab VS Code extension already sends to the
server when it runs the notebook.

Usage, always locally, never on the runtime::

    python -m sartransfer.sync            # pack into every notebook
    python -m sartransfer.sync --check    # exit 1 if a notebook is stale
    python -m sartransfer.sync nb.ipynb   # pack into specific notebooks
    python -m sartransfer.sync --clear    # empty the cells again, before publishing
    python -m sartransfer.sync --published  # exit 1 unless every notebook is a published copy

What travels: `src/sartransfer/**/*.py` and `pyproject.toml`, gzipped and
base64-encoded, a few hundred kB. What never travels: `.env`, anything under `.git`,
`__pycache__`, data, notebooks, results. Secrets in a notebook cell would be
committed to git, which is the whole reason this is an explicit allowlist of
`*.py` rather than "everything under the repo".

The payload is deterministic: sorted names, fixed mtime, no gzip timestamp. An
unchanged source tree produces a byte-identical cell, so `git diff` stays quiet
unless the code really changed.

The tests cell. A notebook that also holds a cell with the tests markers (TESTS_BEGIN)
gets the tests packed into that cell the same way -
tests/*.py with tests/_setup.py, except the tests that read the notebooks, which stay on the PC
(tests/test_notebook*.py and any test whose source holds `REPO / "notebooks"`, the dry runs of
notebooks 03a, 03b, 04 and 05 among them). Next to them what the packed tests read and the sync
cell does not carry - .env.example (the template, every value empty), the vendored LICENSE and
NOTICE.md files under src/sartransfer, by name (`PACKED_BY_NAME`), and docs/licences/*.md.
Nothing else that is not Python travels, so a stray file under src/ cannot end up in a notebook.
The cell has its own fingerprint and names the source fingerprint it was packed with, it is
stale when either changed. Notebooks without that cell are packed exactly as before.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

BEGIN = "# <<<SARTRANSFER-SYNC-BEGIN>>>"
END = "# <<<SARTRANSFER-SYNC-END>>>"
TESTS_BEGIN = "# <<<SARTRANSFER-TESTS-BEGIN>>>"
TESTS_END = "# <<<SARTRANSFER-TESTS-END>>>"

PKG_DIR = Path(__file__).resolve().parent          # .../src/sartransfer
SRC_DIR = PKG_DIR.parent                           # .../src
REPO_ROOT = SRC_DIR.parent                         # repo root
TESTS_DIR = REPO_ROOT / "tests"

EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
NOT_PACKED_TESTS = "test_notebook"                 # the notebook dry runs read the notebooks themselves ...
READS_NOTEBOOKS = 'REPO / "notebooks"'             # ... as does any test with this in its source, it stays on the PC
PACKED_BY_NAME = ("LICENSE", "NOTICE.md")          # the only non-Python files under src/ the tests cell carries
PACKED_ROOT_FILES = (".env.example",)              # the root files the packed tests read, a template without values


# --------------------------------------------------------------------- packing

def _normalised(path: Path) -> bytes:
    """File contents with LF line endings.

    Git rewrites line endings on checkout, so reading raw bytes would make the
    fingerprint change after a clone even though no code did. Normalising here
    keeps `--check` honest across machines.
    """
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")


def collect_sources() -> dict[str, bytes]:
    """Return {archive path: bytes} for everything that should travel."""
    files: dict[str, bytes] = {}
    for p in sorted(PKG_DIR.rglob("*.py")):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        files[f"sartransfer/{p.relative_to(PKG_DIR).as_posix()}"] = _normalised(p)
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        files["pyproject.toml"] = _normalised(pyproject)
    return files


def collect_tests() -> dict[str, bytes]:
    """Return {path from the repo root: bytes} for everything the tests cell carries.

    tests/*.py with tests/_setup.py, except the tests that read the notebooks (tests/test_notebook*.py
    and any test whose source holds READS_NOTEBOOKS), which stay on the PC. Next to them what the
    packed tests read and the sync cell does not carry - the root files PACKED_ROOT_FILES
    (.env.example, a template without values), the vendored LICENSE and NOTICE.md files under
    src/sartransfer, by name, and the licence records docs/licences/*.md. An allowlist, as the
    sync cell is: a `.env`, a weights file or anything else that strays under src/ never travels.
    """
    files: dict[str, bytes] = {}
    for p in sorted(TESTS_DIR.glob("*.py")):
        text = _normalised(p)
        if p.name.startswith(NOT_PACKED_TESTS) or READS_NOTEBOOKS.encode("utf-8") in text:
            continue
        files[f"tests/{p.name}"] = text
    for name in PACKED_ROOT_FILES:
        if (REPO_ROOT / name).is_file():
            files[name] = _normalised(REPO_ROOT / name)
    for p in sorted(PKG_DIR.rglob("*")):
        if p.is_file() and p.name in PACKED_BY_NAME and not any(part in EXCLUDE_DIRS for part in p.parts):
            files[f"src/sartransfer/{p.relative_to(PKG_DIR).as_posix()}"] = _normalised(p)
    for p in sorted((REPO_ROOT / "docs" / "licences").glob("*.md")):
        files[f"docs/licences/{p.name}"] = _normalised(p)
    return files


def make_payload(files: dict[str, bytes]) -> tuple[str, str]:
    """Deterministic gzipped tar, base64 encoded. Returns (payload, sha256)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:        # uncompressed first
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0                                   # no timestamps
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    import gzip

    gz = gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)
    payload = base64.b64encode(gz).decode("ascii")
    return payload, hashlib.sha256(gz).hexdigest()[:16]


def source_fingerprint(files: dict[str, bytes]) -> str:
    """Hash of the source tree itself, independent of archive framing."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update(hashlib.sha256(files[name]).digest())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- cell source

def build_cell(payload: str, sha: str, fingerprint: str, n_files: int) -> str:
    """The cell that unpacks the payload on the Colab runtime."""
    chunks = [payload[i:i + 96] for i in range(0, len(payload), 96)]
    literal = "\n".join(f'    "{c}"' for c in chunks)
    return f'''{BEGIN}
# Generated by `python -m sartransfer.sync` -- do not edit by hand.
# source fingerprint: {fingerprint}   files: {n_files}   payload sha256: {sha}
#
# The package source travels inside this notebook, so the Colab runtime needs no
# git, no clone and no GitHub token. Re-run `python -m sartransfer.sync` locally
# after changing anything under src/, then re-run this cell.
import base64, gzip, hashlib, io, sys, tarfile, subprocess
from pathlib import Path

_PAYLOAD = (
{literal}
)
_EXPECTED_SHA = "{sha}"
_DEST = Path("/content/sartransfer-src")

_gz = base64.b64decode(_PAYLOAD)
_got = hashlib.sha256(_gz).hexdigest()[:16]
assert _got == _EXPECTED_SHA, f"payload corrupted: {{_got}} != {{_EXPECTED_SHA}}"

if _DEST.exists():
    import shutil
    shutil.rmtree(_DEST)
_DEST.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(_gz))) as _tar:
    try:
        _tar.extractall(_DEST, filter="data")   # refuses paths outside _DEST
    except TypeError:
        _tar.extractall(_DEST)                  # an older Python without the filter argument

if str(_DEST) in sys.path:
    sys.path.remove(str(_DEST))
sys.path.insert(0, str(_DEST))
for _m in [m for m in sys.modules if m == "sartransfer" or m.startswith("sartransfer.")]:
    del sys.modules[_m]

# Colab already ships numpy, pandas, pillow and tqdm. Install only what is absent.
for _mod, _pip in [("numpy", "numpy"), ("pandas", "pandas"), ("PIL", "pillow"), ("tqdm", "tqdm")]:
    try:
        __import__(_mod)
    except ImportError:
        print("installing", _pip)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pip], check=True)

import sartransfer
print("sartransfer", sartransfer.__version__, "from", sartransfer.__file__)
print("source fingerprint:", "{fingerprint}", "|", {n_files}, "files")
{END}'''


# The tests cell. A plain template filled by str.replace, so the braces of its code stay as written.
TESTS_TEMPLATE = '''@@BEGIN@@
# Generated by `python -m sartransfer.sync` -- do not edit by hand.
# tests fingerprint @@FINGERPRINT@@   files @@N_FILES@@   payload sha256 @@SHA@@   source fingerprint @@SOURCE@@
#
# The tests travel inside this notebook as the package source does in the sync cell above.
# This cell unpacks them to /content/sartransfer-tests in the layout of the repository -
# tests/, docs/licences/ and src/ with a copy of the source the sync cell unpacked - so
# tests/_setup.py finds src/ next to tests/ as on the PC. It stops when the sync cell above
# unpacked another source than the one these tests were packed with. Re-run
# `python -m sartransfer.sync` locally after changing a test or src/, then both cells again.
import base64
import gzip
import hashlib
import io
import shutil
import tarfile
from pathlib import Path

_TESTS_PAYLOAD = (
@@LITERAL@@
)
_TESTS_SHA = "@@SHA@@"
_PACKED_WITH = "@@SOURCE@@"
_SOURCE = Path("/content/sartransfer-src")
_TESTS_ROOT = Path("/content/sartransfer-tests")

assert (_SOURCE / "sartransfer" / "__init__.py").is_file(), "no source on this runtime, run the sync cell above first"
# the fingerprint of the unpacked source, made as sartransfer.sync.source_fingerprint makes it
_names = sorted([f"sartransfer/{p.relative_to(_SOURCE / 'sartransfer').as_posix()}"
                 for p in (_SOURCE / "sartransfer").rglob("*.py")]
                + (["pyproject.toml"] if (_SOURCE / "pyproject.toml").is_file() else []))
_h = hashlib.sha256()
for _n in _names:
    _h.update(_n.encode())
    _h.update(hashlib.sha256((_SOURCE / _n).read_bytes()).digest())
assert _h.hexdigest()[:16] == _PACKED_WITH, (
    f"the sync cell above unpacked the source {_h.hexdigest()[:16]}, these tests were packed with "
    f"the source {_PACKED_WITH} - run the sync cell again, or python sync.py on the PC and then both cells")

_gz = base64.b64decode(_TESTS_PAYLOAD)
_got = hashlib.sha256(_gz).hexdigest()[:16]
assert _got == _TESTS_SHA, f"tests payload corrupted, {_got} against {_TESTS_SHA}"
if _TESTS_ROOT.exists():
    shutil.rmtree(_TESTS_ROOT)
shutil.copytree(_SOURCE / "sartransfer", _TESTS_ROOT / "src" / "sartransfer",
                ignore=shutil.ignore_patterns("__pycache__"))
if (_SOURCE / "pyproject.toml").is_file():
    shutil.copyfile(_SOURCE / "pyproject.toml", _TESTS_ROOT / "pyproject.toml")
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(_gz))) as _tar:
    try:
        _tar.extractall(_TESTS_ROOT, filter="data")   # refuses paths outside _TESTS_ROOT
    except TypeError:
        _tar.extractall(_TESTS_ROOT)                  # an older Python without the filter argument
print(f"tests unpacked to {_TESTS_ROOT} - @@N_FILES@@ files, "
      f"{len(list((_TESTS_ROOT / 'tests').glob('test_*.py')))} test files, tests fingerprint "
      f"@@FINGERPRINT@@, next to the source {_PACKED_WITH}")
@@END@@'''


def build_tests_cell(payload: str, sha: str, fingerprint: str, n_files: int, source: str) -> str:
    """The cell that unpacks the tests next to a copy of the source on the Colab runtime."""
    chunks = [payload[i:i + 96] for i in range(0, len(payload), 96)]
    literal = "\n".join(f'    "{c}"' for c in chunks)
    filled = {"@@BEGIN@@": TESTS_BEGIN, "@@END@@": TESTS_END, "@@FINGERPRINT@@": fingerprint,
              "@@N_FILES@@": str(n_files), "@@SHA@@": sha, "@@SOURCE@@": source, "@@LITERAL@@": literal}
    text = TESTS_TEMPLATE
    for mark, value in filled.items():
        text = text.replace(mark, value)
    return text


def pack_tests(source: str) -> dict:
    """The tests cell for the source fingerprint `source`, with its numbers."""
    files = collect_tests()
    fingerprint = source_fingerprint(files)
    payload, sha = make_payload(files)
    return {"cell": build_tests_cell(payload, sha, fingerprint, len(files), source),
            "fingerprint": fingerprint, "files": len(files), "kb": len(payload) / 1024}


# ------------------------------------------------------------------- notebooks

def find_notebooks(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    return sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def cell_index(nb: dict, marker: str = BEGIN) -> int | None:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return i
    return None


def tests_cell_index(nb: dict) -> int | None:
    return cell_index(nb, TESTS_BEGIN)


def embedded_fingerprint(nb: dict) -> str | None:
    i = cell_index(nb)
    if i is None:
        return None
    for line in "".join(nb["cells"][i]["source"]).splitlines():
        if line.startswith("# source fingerprint:"):
            return line.split()[3]
    return None


def embedded_tests_fingerprint(nb: dict) -> tuple[str, str] | None:
    """(tests fingerprint, source fingerprint) of the tests cell, None without one."""
    i = tests_cell_index(nb)
    if i is None:
        return None
    for line in "".join(nb["cells"][i]["source"]).splitlines():
        found = re.match(r"# tests fingerprint (\w+) .* source fingerprint (\w+)$", line)
        if found:
            return found.group(1), found.group(2)
    return None


def _put(nb: dict, i: int, source: str) -> None:
    nb["cells"][i]["source"] = source.splitlines(True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None


def sync(paths: list[Path], check_only: bool = False) -> int:
    files = collect_sources()
    fingerprint = source_fingerprint(files)
    payload, sha = make_payload(files)
    cell_src = build_cell(payload, sha, fingerprint, len(files))
    tests = None                                  # packed only when a notebook has a tests cell

    stale, missing = [], []
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        i = cell_index(nb)
        if i is None:
            missing.append(p)
            continue
        j = tests_cell_index(nb)
        old = [] if embedded_fingerprint(nb) == fingerprint else ["sync cell"]
        if j is not None:
            tests = tests or pack_tests(fingerprint)
            if embedded_tests_fingerprint(nb) != (tests["fingerprint"], fingerprint):
                old.append("tests cell")
        if not old:
            print(f"up to date  {p.name}")
            continue
        stale.append(p.name + (f" ({', '.join(old)})" if j is not None else ""))
        if check_only:
            continue
        if "sync cell" in old:
            _put(nb, i, cell_src)
        if "tests cell" in old:
            _put(nb, j, tests["cell"])
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"updated     {stale[-1]}")

    for p in missing:
        print(f"NO SYNC CELL  {p.name} -- add one containing {BEGIN}")

    print(f"\n{len(files)} files | fingerprint {fingerprint} | payload {len(payload)/1024:.1f} kB")
    if tests is not None:
        print(f"{tests['files']} files in the tests cell | tests fingerprint {tests['fingerprint']} | "
              f"payload {tests['kb']:.1f} kB")

    if check_only and (stale or missing):
        print("\nstale:", ", ".join(stale) or "none")
        print("run: python -m sartransfer.sync")
        return 1
    return 0


CLEARED = f"""{BEGIN}
# Empty sync cell: published notebooks carry no packed code. On your own machine,
# in the repository folder, run   python sync.py   and open the notebook again;
# this cell then holds the package source and unpacks it on the runtime.
raise RuntimeError("The sync cell is empty. On your own machine, in the repository folder, "
                   "run:  python sync.py  and reopen this notebook.")
{END}"""

TESTS_CLEARED = f"""{TESTS_BEGIN}
# Empty tests cell - published notebooks carry no packed code. On your own machine, in the
# repository folder, run   python sync.py   and open the notebook again, this cell then
# holds the tests and unpacks them on the runtime.
raise RuntimeError("The tests cell is empty. On your own machine, in the repository folder, "
                   "run python sync.py and reopen this notebook.")
{TESTS_END}"""


def is_cleared(nb: dict) -> bool:
    """True when the notebook's sync cell is the empty stub of `clear` (a published copy)."""
    i = cell_index(nb)
    return i is not None and "".join(nb["cells"][i]["source"]).strip() == CLEARED.strip()


def published_check(paths: list[Path]) -> int:
    """Exit 1 unless every notebook is a published copy, changing nothing.

    A published copy has the empty stub of `clear` in its sync cell and in its tests cell when it has
    one, and no code cell with outputs or an execution count - so no packed code and nothing a run
    printed is committed. The CI runs this on every commit; run it before a push (2026-10-04).
    """
    bad = []
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        if cell_index(nb) is None:
            bad.append(f"{p.name}: no sync cell")
        elif not is_cleared(nb):
            bad.append(f"{p.name}: the sync cell holds packed code, run   python sync.py --clear")
        j = tests_cell_index(nb)
        if j is not None and "".join(nb["cells"][j]["source"]).strip() != TESTS_CLEARED.strip():
            bad.append(f"{p.name}: the tests cell holds packed tests, run   python sync.py --clear")
        run = [k for k, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
               and (c.get("outputs") or c.get("execution_count") is not None)]
        if run:
            bad.append(f"{p.name}: {len(run)} code cell(s) with outputs or an execution count, clear them first")
    for line in bad:
        print("NOT PUBLISHED  " + line)
    if bad:
        return 1
    print(f"published   {len(paths)} notebooks - every sync cell cleared, no outputs, no execution counts")
    return 0


def clear(paths: list[Path]) -> int:
    """Replace every sync cell's payload, and every tests cell's, by a short stub that explains how to fill it."""
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        i = cell_index(nb)
        if i is None:
            print(f"NO SYNC CELL  {p.name}")
            continue
        _put(nb, i, CLEARED)
        j = tests_cell_index(nb)
        if j is not None:
            _put(nb, j, TESTS_CLEARED)
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"cleared     {p.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="*", help="default: every notebook in notebooks/")
    ap.add_argument("--check", action="store_true",
                    help="report stale notebooks and exit 1, changing nothing")
    ap.add_argument("--clear", action="store_true",
                    help="empty the sync cells (no packed code), e.g. before publishing")
    ap.add_argument("--published", action="store_true",
                    help="exit 1 unless every notebook is a published copy - cleared cells, no outputs")
    args = ap.parse_args(argv)
    if args.clear:
        return clear(find_notebooks(args.notebooks))
    if args.published:
        return published_check(find_notebooks(args.notebooks))
    return sync(find_notebooks(args.notebooks), check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
