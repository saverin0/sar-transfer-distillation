"""env.load_env reads every .env in search order and merges them, the first file wins per key."""
import _setup  # noqa: F401

import os

from sartransfer import env

base = _setup.TMP_BASE / "env"
base.mkdir(parents=True, exist_ok=True)
first, second = base / "first.env", base / "second.env"
first.write_text("HF_WRITE_TOKEN=write_from_first\n", encoding="utf-8")
second.write_text("HF_TOKEN=read_from_second\nHF_WRITE_TOKEN=write_from_second\n", encoding="utf-8")
for k in ("HF_TOKEN", "HF_WRITE_TOKEN"):
    os.environ.pop(k, None)
old = env.SEARCH_PATHS
env.SEARCH_PATHS = (base / "missing.env", first, second)
try:
    have = env.load_env(verbose=False)
finally:
    env.SEARCH_PATHS = old
assert os.environ["HF_TOKEN"] == "read_from_second", "a key only in the second file was not loaded"
assert os.environ["HF_WRITE_TOKEN"] == "write_from_first", "the first file must win per key"
assert set(have) >= {"HF_TOKEN", "HF_WRITE_TOKEN"}

# an environment variable set beforehand is never overwritten
os.environ["HF_TOKEN"] = "already_there"
env.SEARCH_PATHS = (second,)
try:
    env.load_env(verbose=False)
finally:
    env.SEARCH_PATHS = old
assert os.environ["HF_TOKEN"] == "already_there"
for k in ("HF_TOKEN", "HF_WRITE_TOKEN"):
    os.environ.pop(k, None)

# ---- the data settings of part four, from the same file -------------
from sartransfer.domains import hub  # noqa: E402

assert {"DATA_HF_REPO", "DATA_DRIVE_DIR"} <= set(env.KNOWN)
assert "DATA_HF_REPO" in env.__doc__ and "DATA_DRIVE_DIR" in env.__doc__
for k in ("DATA_HF_REPO", "DATA_DRIVE_DIR"):
    os.environ.pop(k, None)
# unset = a clear error that names the key and the .env file, no default is guessed
for fn in (hub.data_repo_id, hub.pilot_repo_id):
    try:
        fn()
        raise SystemExit(f"{fn.__name__} gave an id without DATA_HF_REPO")
    except ValueError as err:
        assert "DATA_HF_REPO" in str(err) and ".env" in str(err), err
third = base / "third.env"
third.write_text("DATA_HF_REPO=an-owner/a-data-repo\nDATA_DRIVE_DIR=a-folder-under-mydrive\n", encoding="utf-8")
env.SEARCH_PATHS = (third,)
try:
    have = env.load_env(verbose=False)
finally:
    env.SEARCH_PATHS = old
assert set(have) >= {"DATA_HF_REPO", "DATA_DRIVE_DIR"}
assert hub.data_repo_id() == "an-owner/a-data-repo" == hub.pilot_repo_id()
assert env.require("DATA_DRIVE_DIR") == "a-folder-under-mydrive"
try:                                   # an owner is no longer accepted, the id is the full owner/name
    hub.pilot_repo_id("someone")
    raise SystemExit("pilot_repo_id accepted an owner")
except TypeError:
    pass
for bad in ("no-slash", "a/b/c", "/name", "owner/", " "):
    os.environ["DATA_HF_REPO"] = bad
    try:
        hub.data_repo_id()
        raise SystemExit(f"data_repo_id accepted {bad!r}")
    except ValueError:
        pass
for k in ("DATA_HF_REPO", "DATA_DRIVE_DIR"):
    os.environ.pop(k, None)

# ---- the settings of notebook 03 (2026-10-04) - the owner of the SAR dataset copies and part two's folder
NEW = ("DATASET_HF_OWNER", "REFERENCE_DRIVE_DIR")
assert set(NEW) <= set(env.KNOWN) and all(k in env.__doc__ for k in NEW)
example = (_setup.REPO / ".env.example").read_text(encoding="utf-8").splitlines()
for k in NEW:
    at = example.index(f"{k}=")                                    # an empty line to fill in ...
    assert example[at - 1].startswith("# "), f"{k} has no comment line in .env.example"   # ... with its comment
had = {k: os.environ.pop(k, None) for k in NEW}
fourth = base / "fourth.env"
fourth.write_text("DATASET_HF_OWNER=an-owner\nREFERENCE_DRIVE_DIR=a-reference-folder\n", encoding="utf-8")
env.SEARCH_PATHS = (fourth,)
try:
    have = env.load_env(verbose=False)
finally:
    env.SEARCH_PATHS = old
assert set(have) >= set(NEW) and os.environ["DATASET_HF_OWNER"] == "an-owner"
assert env.require("REFERENCE_DRIVE_DIR") == "a-reference-folder"
assert hub.repo_id("snow", os.environ["DATASET_HF_OWNER"]) == "an-owner/sar-transfer-snow"
for k, v in had.items():
    os.environ.pop(k, None)
    if v is not None:
        os.environ[k] = v

# ---- the model cache folder from the .env (2026-10-04), only one .env is read on Drive -------
on_drive = [p for p in env.SEARCH_PATHS if env.MYDRIVE.parent in p.parents]   # the paths under the Drive mount
assert on_drive == [env.MYDRIVE / ".env"], env.SEARCH_PATHS
assert "MODEL_CACHE_DRIVE_DIR" in env.KNOWN and "MODEL_CACHE_DRIVE_DIR" in env.__doc__
example = (_setup.REPO / ".env.example").read_text(encoding="utf-8").splitlines()
assert example[example.index("MODEL_CACHE_DRIVE_DIR=") - 1].startswith("# ")
CACHE_KEYS = ("MODEL_CACHE_DRIVE_DIR", "HF_HOME", "HF_HUB_DISABLE_SYMLINKS_WARNING")
had_cache = {k: os.environ.get(k) for k in CACHE_KEYS}
os.environ.pop("MODEL_CACHE_DRIVE_DIR", None)
mydrive = base / "MyDrive"
real_ismount = os.path.ismount
os.path.ismount = lambda path: path == "/content/drive"
try:
    assert env.model_cache_on_drive(mount="/not/mounted") is None, "no Drive, no cache"
    env.SEARCH_PATHS = (base / "missing.env",)
    try:
        env.model_cache_on_drive(mydrive=mydrive)
        raise SystemExit("a missing MODEL_CACHE_DRIVE_DIR gave a cache folder")
    except RuntimeError as err:
        assert "MODEL_CACHE_DRIVE_DIR is not in the .env" in str(err), err
    fifth = base / "fifth.env"
    for bad in ("/content/drive/MyDrive/shared/weights", "MyDrive/shared/weights", "/shared/weights", "D:/weights"):
        fifth.write_text(f"MODEL_CACHE_DRIVE_DIR={bad}\n", encoding="utf-8")
        env.SEARCH_PATHS = (fifth,)
        os.environ.pop("MODEL_CACHE_DRIVE_DIR", None)
        try:
            env.model_cache_on_drive(mydrive=mydrive)
            raise SystemExit(f"MODEL_CACHE_DRIVE_DIR={bad} gave a cache folder")
        except RuntimeError as err:
            assert "without the MyDrive prefix" in str(err) and bad in str(err), err
    assert not (mydrive / "content").exists() and not (mydrive / "MyDrive").exists(), "a wrong folder was made"
    fifth.write_text("MODEL_CACHE_DRIVE_DIR=shared/weights/\n", encoding="utf-8")
    os.environ.pop("MODEL_CACHE_DRIVE_DIR", None)
    root = env.model_cache_on_drive(mydrive=mydrive)                 # the key is read from the .env here
    assert root == mydrive / "shared" / "weights" and root.is_dir() and os.environ["HF_HOME"] == str(root)
    assert env.model_cache_on_drive(root=base / "given", mydrive=mydrive) == base / "given"
finally:
    os.path.ismount = real_ismount
    env.SEARCH_PATHS = old
    for k, v in had_cache.items():                                    # every key the function touches, as it was
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print("test_env: all checks passed")
