"""What may not be in the files of this repository, so a public copy holds no leftovers.

Every text file (the tracked and the new files when this is a git checkout, else every file under the
folder) and the text of every notebook cell, the packed sync and tests cells left out (base64).
- no pointer to a private-only file, and no private name (both lists live in a private-only file of the
  repository, read through tests/_setup.py, so a public copy holds neither the lists nor the names)
- no internal rule numbers, in any case
- no path of a local machine's folders
- in a public copy every sync cell is cleared - no packed code, no fingerprint
"""
import _setup  # noqa: F401

import json
import re
import subprocess

from sartransfer import sync as SYNC

ROOT = _setup.REPO
PRIVATE_ONLY, POINTERS = _setup.private_only()   # both empty in a public copy
OTHER = _setup.private_names()
RULE = re.compile(r"\brules? \d+\b", re.IGNORECASE)
LOCAL = re.compile(r"\b[a-z]:[\\/](?:users|claude|data|projects|envs)\b", re.IGNORECASE)
TEXT = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml", ".example", ".ipynb", ""}


def files():
    """The files to read, as paths relative to the repository root."""
    if (ROOT / ".git").exists():
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
        names = [n for n in out if n]
    else:                                                # an unpacked copy, no git
        names = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and "weights" not in p.parts]
    return [n for n in names if not n.startswith("weights/") and n not in PRIVATE_ONLY
            and (ROOT / n).suffix in TEXT and (ROOT / n).is_file()]


def texts(name):
    """(where, text) of a file, a notebook cell by cell without the packed cells."""
    raw = (ROOT / name).read_text(encoding="utf-8", errors="replace")
    if not name.endswith(".ipynb"):
        return [(name, raw)]
    out = []
    for k, cell in enumerate(json.loads(raw)["cells"]):
        src = "".join(cell["source"])
        if "SARTRANSFER-SYNC-BEGIN" in src or "SARTRANSFER-TESTS-BEGIN" in src:
            continue
        out.append((f"{name} cell {cell.get('id', k)}", src))
    return out


# the export-ignore list of the attributes file is the private-only list
attributes = ROOT / ("." + "git" + "attributes")
if attributes.is_file() and PRIVATE_ONLY:
    ignored = {line.split()[0] for line in attributes.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#") and "export-ignore" in line}
    assert ignored == set(PRIVATE_ONLY), f"the export-ignore list differs, {sorted(ignored ^ set(PRIVATE_ONLY))}"

found, read = [], 0
names = files()
for name in names:
    for where, text in texts(name):
        read += 1
        for line_no, line in enumerate(text.splitlines(), 1):
            hits = [w for w in (*POINTERS, *OTHER) if w in line]
            hits += [m.group(0) for m in RULE.finditer(line)] + [m.group(0) for m in LOCAL.finditer(line)]
            if hits:
                found.append(f"{where}:{line_no} {hits} | {line.strip()[:120]}")
assert read > 20, f"only {read} texts were read, the file list is wrong"
assert not found, "leftovers that may not be published\n" + "\n".join(found)

# a public copy carries no packed code
public = _setup.is_public_copy()
notebooks = [n for n in names if n.endswith(".ipynb")]
if public:
    packed = [n for n in notebooks if not SYNC.is_cleared(json.loads((ROOT / n).read_text(encoding="utf-8")))]
    assert not packed, f"a public copy with packed sync cells, run   python sync.py --clear   first - {packed}"
print(f"test_hygiene - {read} texts read, nothing points to a private-only file, no rule number, no local path, no "
      f"other repository, {'a public copy, every sync cell cleared' if public else 'the private repository'}")
print("test_hygiene: all checks passed")
