"""Read secrets and the data settings from a .env file into os.environ, with a getpass fallback.

The file is looked for in the places of SEARCH_PATHS, in order, and the first file that holds a key
wins. Values already present in os.environ always win, so an env var set by hand is never silently
overwritten by a stale file. Nothing here ever prints a secret.

Besides the tokens the file says where the data is, so that no notebook and no module of this
repository spells out a folder or a repository (2026-10-03)

    DATA_HF_REPO    the private Hugging Face dataset repo of the pilot data, the full "owner/name"
                    (domains.hub.data_repo_id reads it)
    DATA_DRIVE_DIR  the folder under MyDrive that holds pilot/ and stage0/ of the data, without
                    the MyDrive prefix (the notebooks build their Drive paths from it). Its
                    results/ holds the saved rows of the part three pilot, which notebook 03
                    reads and never writes
    DATASET_HF_OWNER     the Hugging Face user or organisation that holds the private copies of
                         the four prepared SAR datasets, <owner>/sar-transfer-<name>
                         (domains.hub.repo_id). Optional - unset, notebook 03 copies them from
                         Drive, MyDrive/datasets/sar-transfer-<name>/ (2026-10-04)
    REFERENCE_DRIVE_DIR  the folder under MyDrive that holds part two's results/, the saved
                         rows of the references and of the control run, without the MyDrive
                         prefix. Notebook 03 reads it and never writes there (2026-10-04)
    MODEL_CACHE_DRIVE_DIR  the folder under MyDrive of the model weights shared by every project
                         (HF_HOME, `model_cache_on_drive`), without the MyDrive prefix (2026-10-04)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

MYDRIVE = Path("/content/drive/MyDrive")
SEARCH_PATHS = (
    MYDRIVE / ".env",                                   # Colab with Drive mounted
    Path("/content/.env"),                              # Colab, this runtime only
    Path.cwd() / ".env",
    Path(__file__).resolve().parents[2] / ".env",       # repo root, local runs
)

# Every key this repository reads, and nothing else - load_env prints each as set or MISSING, so a key
# listed here is one the user is asked to put into the .env file (2026-10-04).
KNOWN = ("HF_TOKEN", "HF_WRITE_TOKEN", "DATA_HF_REPO", "DATA_DRIVE_DIR",
         "DATASET_HF_OWNER", "REFERENCE_DRIVE_DIR", "MODEL_CACHE_DRIVE_DIR")


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser. Ignores blanks, # comments and empty values.

    Rejects a value that swallowed the following line: a .env that loses a line
    break yields `HF_TOKEN=<token>DATA_HF_REPO=owner/name`, a token that is wrong
    and a setting that is lost. Any `NAME=` inside a value counts (NAME: a capital,
    then two or more capitals, digits or underscores), except at the very end or
    before another `=` (base64 padding). Better to drop it loudly than pass it on.
    """
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if not val:
            continue
        merged = [k for k in KNOWN if k != key and k + "=" in val]
        if merged or re.search(r"[A-Z][A-Z0-9_]{2,}=(?!=|$)", val):   # never prints the value
            print(f"  SKIPPING {key} from {path}: value also contains {merged or 'another NAME=... entry'} --"
                  " the file is missing a line break between entries")
            continue
        out[key] = val
    return out


def load_env(path: str | Path | None = None, verbose: bool = True) -> list[str]:
    """Load every .env found into os.environ, in search order. Returns the keys now set.

    Existing environment variables are left alone, and a key set by an earlier file is left
    alone by a later one (the first file wins per key).
    """
    candidates = [Path(path)] if path else list(SEARCH_PATHS)
    used: list[Path] = []
    for p in candidates:
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        for key, val in parse_env_file(p).items():
            os.environ.setdefault(key, val)
        used.append(p)

    if used and verbose:
        print("loaded secrets from", ", ".join(str(p) for p in used))
    elif verbose:
        print("no .env found in:", ", ".join(str(p) for p in candidates))

    have = [k for k in KNOWN if os.environ.get(k)]
    if verbose:
        for k in KNOWN:
            print(f"  {k:<19} {'set' if os.environ.get(k) else 'MISSING'}")
    return have


def require(key: str, why: str = "") -> str:
    """Return a secret, prompting with getpass if the .env did not supply it."""
    val = os.environ.get(key)
    if not val:
        from getpass import getpass

        val = getpass(f"{key}{' for ' + why if why else ''}: ")
        os.environ[key] = val
    return val


def model_cache_on_drive(root: str | Path | None = None, mydrive: str | Path = MYDRIVE,
                         mount: str = "/content/drive") -> Path | None:
    """Keep downloaded model weights on Drive instead of the runtime's disk.

    Every teacher's weights (DINOv3, TerraFM, SigLIP2, PE-Spatial) come through
    the Hugging Face hub, which caches under HF_HOME. Pointing that at Drive
    means each model is downloaded ONCE and reused across sessions. Must be
    called before transformers / huggingface_hub are imported. The folder is
    `root`, else MyDrive / MODEL_CACHE_DRIVE_DIR from the .env (read here when it is
    not in the environment yet); a missing key is an error that names it. Returns
    the cache path, or None (with a message) if Drive is not mounted.
    """
    if not os.path.ismount(mount):
        print("Drive not mounted -- model cache stays on the runtime (re-downloaded next session)")
        return None
    if root is None:
        if not os.environ.get("MODEL_CACHE_DRIVE_DIR"):
            load_env(verbose=False)
        folder = os.environ.get("MODEL_CACHE_DRIVE_DIR", "").strip().rstrip("/")
        if not folder:
            raise RuntimeError("MODEL_CACHE_DRIVE_DIR is not in the .env - the folder under MyDrive of the model "
                               "weights shared by every project, without the MyDrive prefix")
        if folder.startswith(("/", "\\", "MyDrive", "content/")) or ":" in folder:
            raise RuntimeError("MODEL_CACHE_DRIVE_DIR must be a folder under MyDrive without the MyDrive prefix, "
                               f"for example models/hf_cache, got {folder!r}")
        root = Path(mydrive) / folder
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # Drive has no symlinks; hub copies instead
    have = sorted(d.name for d in (root / "hub").glob("models--*")) if (root / "hub").exists() else []
    print(f"model cache: {root} | cached: {have or 'nothing yet'}")
    return root
