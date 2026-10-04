"""The Hugging Face account behind the tokens, checked before the pilot data is uploaded.

A notebook runs `check` right after its setup cell. It prints, never
a token
- with the read token, the user name and the account type from whoami, and whether the
  token can read the gated DINOv3 file (a HEAD on its config.json, 200 = yes, 401 or 403 = no)
- with the write token, when there is one, the same user name and whether the token has
  write rights (the role of a classic token, or the permissions of a fine-grained one)
- the user's private dataset and model repos with their sizes, the total in GB next to the
  documented private storage of the account type, the space the pilot needs, and a verdict
- the free space on Drive

Every network error is printed as words, nothing stops the notebook. A token goes to
addresses under `majortom.HF` only, as in majortom, through a session that reads no netrc
file. Every printed text goes through `hide`, which takes the tokens out of error messages.

The quota. Read on 2026-09-30 on https://huggingface.co/docs/hub/storage-limits, table
"Storage plans" - private storage is 100 GB for a free user or org, 1 TB plus pay as you go
for PRO, public storage is best effort for free users. Part two's private copies hold about
31 GB, the pilot needs about 55 to 70 GB (`pilot_need_gb`), so a free account may be over.

The space the pilot needs is computed here from the constants of pilotdata and aerial, the
notebook retypes nothing. A pair file has `PAIR_FILE_MB` on disk, the stage 0 run measured
it (500 cells, 2026-09-30). The aerial sets are uncompressed npz files of 8 bit arrays.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Mapping

import requests

from .majortom import HF

QUOTA_URL = "https://huggingface.co/docs/hub/storage-limits"
QUOTA_READ = "2026-09-30"
QUOTA_TABLE = "Storage plans"
PRIVATE_QUOTA_GB = {"free": 100.0, "pro": 1000.0}    # private storage per account type
GATED_REPO = "facebook/dinov3-vitl16-pretrain-sat493m"   # teachers.dinov3_sat.HF_ID, gated
GATED_FILE = "config.json"
PAIR_FILE_MB = 2.38          # a pair file on disk, measured in stage 0 (384 px, native bands)
TIMEOUT_S = 30.0
MB = 1e6
GB = 1e9


def hide(text: Any, tokens: Iterable[str | None]) -> str:
    """The text without the tokens."""
    out = str(text)
    for token in tokens:
        if token:
            out = out.replace(token, "<token>")
    return out


def api_for(token: str):
    """An HfApi that holds the token. Tests replace this with a stand-in."""
    from huggingface_hub import HfApi

    return HfApi(token=token)


def account_type(answer: Mapping) -> str:
    """"pro", "free" or "unknown" from a whoami answer (its isPro entry)."""
    pro = answer.get("isPro")
    if pro is None:
        return "unknown"
    return "pro" if pro else "free"


def write_rights(answer: Mapping) -> tuple[str, str]:
    """Whether the token of a whoami answer may write, ("yes" | "no" | "unknown", why).

    A classic token carries auth.accessToken.role, "read" or "write". A fine-grained token
    carries role "fineGrained" and its permissions under auth.accessToken.fineGrained, a
    global list and scoped lists per entity. Writing to the user's own repos needs
    "repo.write" in the global list or in the list scoped to the user.
    """
    access = (answer.get("auth") or {}).get("accessToken") or {}
    role = access.get("role")
    if role == "write":
        return "yes", "role write"
    if role == "read":
        return "no", "role read"
    if role == "fineGrained":
        fine = access.get("fineGrained") or {}
        allowed = list(fine.get("global") or [])
        for scope in fine.get("scoped") or []:
            entity = (scope.get("entity") or {}).get("name")
            if entity in (None, answer.get("name")):
                allowed += list(scope.get("permissions") or [])
        if "repo.write" in allowed:
            return "yes", "fine-grained token with repo.write"
        return "no", "fine-grained token without repo.write for the user"
    return "unknown", f"the answer holds the role {role!r}"


def gated_read(token: str | None, url: str | None = None) -> tuple[str, str]:
    """Whether `token` can read the gated DINOv3 file, ("yes" | "no" | "unclear", detail).

    One HEAD on the file. 200 to 399 = yes (a redirect means the Hub granted the read),
    401 or 403 = no, anything else or no answer = unclear. The token goes only to an address
    under `HF`.
    """
    url = url or f"{HF}/{GATED_REPO}/resolve/main/{GATED_FILE}"
    headers = {}
    if token and url.startswith(HF + "/"):
        headers["Authorization"] = f"Bearer {token}"
    session = requests.Session()
    session.trust_env = False                    # no netrc login, no proxy from the environment
    try:
        with session.head(url, headers=headers, timeout=TIMEOUT_S, allow_redirects=False) as r:
            code = r.status_code
    except requests.RequestException as err:
        return "unclear", f"no answer, {type(err).__name__}"
    finally:
        session.close()
    if 200 <= code < 400:
        return "yes", f"HTTP {code}"
    if code in (401, 403):
        return "no", f"HTTP {code}, the token has no access, accept the licence on {HF}/{GATED_REPO}"
    return "unclear", f"HTTP {code}"


def private_repos(api, user: str) -> list[dict]:
    """The private dataset and model repos of `user` with their bytes, from the file trees."""
    out = []
    for kind, lister in (("dataset", api.list_datasets), ("model", api.list_models)):
        for info in lister(author=user):
            if not getattr(info, "private", None):
                continue
            files = [f for f in api.list_repo_tree(info.id, repo_type=kind, recursive=True)
                     if getattr(f, "size", None) is not None]
            out.append({"repo": info.id, "type": kind, "files": len(files),
                        "bytes": int(sum(f.size for f in files))})
    return out


def pilot_need_gb() -> tuple[float, float]:
    """The space the pilot takes, low and high, in GB.

    Low = the pairs the pilot asks for, high = every drawn cell as a pair, each pair a file of
    `PAIR_FILE_MB`. Plus the aerial sets, 8 bit images with 4 bands and the labels of the
    labelled sets, at their size after halving, uncompressed.
    """
    from . import aerial, pilotdata

    # the held-out patches are drawn out of the distil set, so they are not counted twice
    patches = sum(aerial.SET_SIZES.values()) - aerial.SET_SIZES["distil_heldout"]
    images = patches * len(aerial.BANDS)
    labels = sum(aerial.SET_SIZES[name] for name in aerial.LABELLED)
    aerial_bytes = (images + labels) * aerial.OUT_PX ** 2
    low = pilotdata.N_PAIRS * PAIR_FILE_MB * MB + aerial_bytes
    high = pilotdata.N_PILOT * PAIR_FILE_MB * MB + aerial_bytes
    return low / GB, high / GB


def verdict(used_gb: float, quota_gb: float, need: tuple[float, float]) -> str:
    """"fits" or "does not fit, about X GB over", from the high end of the need."""
    free = quota_gb - used_gb
    low, high = need
    if high <= free:
        return f"fits, the pilot needs about {low:.0f} to {high:.0f} GB, {free:.0f} GB are free"
    return (f"does not fit, about {high - free:.0f} GB over, the pilot needs about {low:.0f} to "
            f"{high:.0f} GB, {free:.0f} GB are free")


def check(read_token: str, write_token: str | None = None, drive_folder: str | os.PathLike | None = None,
          api: Callable[[str], Any] | None = None, say: Callable[[str], Any] = print) -> dict:
    """Print the account check and return it as a dict, without the tokens.

    read_token     HF_TOKEN, the whoami, the gated file and the repo listing use it
    write_token    HF_WRITE_TOKEN or None, its whoami says whether it may write
    drive_folder   a folder on Drive, its free space is printed
    api            a function token -> HfApi, `api_for` by default
    """
    api = api or api_for
    tokens = (read_token, write_token)
    report: dict[str, Any] = {"user": None, "account_type": "unknown", "gated_read": "unclear",
                              "write_token": "missing", "write_rights": "unknown", "repos": [],
                              "used_gb": None, "quota_gb": None, "need_gb": pilot_need_gb(),
                              "verdict": "unknown", "drive_free_gb": None}

    def words(err: BaseException) -> str:
        return f"{type(err).__name__}, {hide(err, tokens)[:200]}"

    say("read token, HF_TOKEN")
    try:
        me = api(read_token).whoami()
        report["user"], report["account_type"] = me.get("name"), account_type(me)
        say(f"  user            {report['user']} ({me.get('type', 'type unknown')}), "
            f"account type {report['account_type']}")
    except Exception as err:
        say(f"  user            could not be read, {words(err)}")
    report["gated_read"], detail = gated_read(read_token)
    say(f"  gated DINOv3    {report['gated_read']}, {detail} on {GATED_REPO}/{GATED_FILE}")

    say("write token, HF_WRITE_TOKEN")
    if not write_token:
        say("  not in the .env file. The upload cell of notebook 01 needs it, put a write token "
            "into the .env file as HF_WRITE_TOKEN before that cell")
    else:
        report["write_token"] = "given"
        try:
            me = api(write_token).whoami()
            report["write_rights"], why = write_rights(me)
            say(f"  user            {me.get('name')}, write rights {report['write_rights']} ({why})")
        except Exception as err:
            say(f"  user            could not be read, {words(err)}")

    say(f"private repos of {report['user'] or 'the user'} (the read token)")
    used = None
    if report["user"]:
        try:
            report["repos"] = private_repos(api(read_token), report["user"])
            for repo in report["repos"]:
                say(f"  {repo['type']:<8} {repo['repo']:<44} {repo['bytes'] / GB:8.2f} GB  "
                    f"{repo['files']} files")
            used = sum(repo["bytes"] for repo in report["repos"]) / GB
            report["used_gb"] = used
            say(f"  total {used:.2f} GB in {len(report['repos'])} private repos")
        except Exception as err:
            say(f"  could not be listed, {words(err)}")
    else:
        say("  not listed, the user is not known")
    plan = report["account_type"]
    quota = PRIVATE_QUOTA_GB.get(plan, PRIVATE_QUOTA_GB["free"])
    report["quota_gb"] = quota
    say(f"  private storage of a {plan if plan != 'unknown' else 'free'} account {quota:,.0f} GB "
        f"({QUOTA_URL}, table {QUOTA_TABLE}, read {QUOTA_READ}, free 100 GB, PRO 1 TB plus pay "
        f"as you go)" + (", the account type is unknown so the free quota is taken" if plan == "unknown" else ""))
    low, high = report["need_gb"]
    say(f"  the pilot needs about {low:.0f} to {high:.0f} GB, pairs x {PAIR_FILE_MB} MB plus the aerial sets")
    if used is not None:
        report["verdict"] = verdict(used, quota, report["need_gb"])
    else:
        report["verdict"] = "unknown, the private repos could not be listed"
    say(f"  verdict         {report['verdict']}")

    if drive_folder is not None:
        from .stage0 import runtime_report

        disk = runtime_report([drive_folder])["disk"][str(drive_folder)]
        report["drive_free_gb"] = disk["free_gb"]
        if disk["free_gb"] is None:
            say(f"Drive  {drive_folder} cannot be read")
        else:
            say(f"Drive  free {disk['free_gb']:.1f} GB of {disk['total_gb']:.1f} GB in {drive_folder}")
    return report
