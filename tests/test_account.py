"""Offline check of distil/account.py, the account check.

Stand-ins for the HfApi calls (whoami, list_datasets, list_models, list_repo_tree), a server on
this machine for the HEAD on the gated file. No network. A token must reach the local server
only when it stands for huggingface.co, and must never be printed.
"""
import _setup  # noqa: F401

import contextlib
import http.server
import io
import shutil
import socket
import threading

from _setup import TMP_BASE  # noqa: E402

from sartransfer.distil import account as AC  # noqa: E402
from sartransfer.distil import aerial as A  # noqa: E402
from sartransfer.distil import majortom as M  # noqa: E402
from sartransfer.distil import pilotdata as D  # noqa: E402
from sartransfer.distil.teachers import dinov3_sat as DS  # noqa: E402

TMP = TMP_BASE / "account"
TMP.mkdir(parents=True, exist_ok=True)
READ, WRITE = "MADE-UP-READ-TOKEN-OF-THE-TEST", "MADE-UP-WRITE-TOKEN-OF-THE-TEST"
GB = AC.GB


def fails(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in the message {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


# ---- the server for the HEAD ------------------------------------------------------------------
HITS = []                                        # (path, headers) of every request
CODE = {"head": 200}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_HEAD(self):
        HITS.append((self.path, dict(self.headers)))
        self.send_response(CODE["head"])
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_HEAD


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"
NETWORK = []
REAL_CONNECT = socket.socket.connect


def connect_here_only(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "::1", "localhost"):
        NETWORK.append(str(host))
        raise ConnectionRefusedError(f"the test must not use the network, asked for {host}")
    return REAL_CONNECT(self, address)


socket.socket.connect = connect_here_only


# ---- the stand-in HfApi -----------------------------------------------------------------------
class Info:
    def __init__(self, id, private):
        self.id, self.private = id, private


class Entry:
    def __init__(self, path, size=None):
        self.path = path
        if size is not None:
            self.size = size


class FakeApi:
    """The four calls of account.py over a made-up account."""

    def __init__(self, token, answers, repos, trees, fail=None):
        self.token, self.answers, self.repos, self.trees, self.fail = token, answers, repos, trees, fail
        assert token in (READ, WRITE), f"an unknown token reached the api, {token[:4]}"

    def whoami(self):
        if self.fail == "whoami":
            raise ConnectionError(f"stand-in, no route to the Hub, the token was {self.token}")
        return self.answers[self.token]

    def list_datasets(self, author):
        if self.fail == "list":
            raise OSError(f"stand-in, the listing failed, the token was {self.token}")
        return [Info(*r) for r in self.repos.get(("dataset", author), [])]

    def list_models(self, author):
        return [Info(*r) for r in self.repos.get(("model", author), [])]

    def list_repo_tree(self, repo_id, repo_type, recursive):
        assert recursive is True and repo_type in ("dataset", "model")
        return [Entry(*e) for e in self.trees[repo_id]]


USER = {"type": "user", "name": "someone", "isPro": False,
        "auth": {"type": "access_token", "accessToken": {"displayName": "read", "role": "read"}}}
WRITER = {"type": "user", "name": "someone", "isPro": False,
          "auth": {"type": "access_token", "accessToken": {"displayName": "w", "role": "write"}}}
REPOS = {("dataset", "someone"): [("someone/sar-transfer-snow", True), ("someone/public-thing", False),
                                  ("someone/sar-transfer-forty-v1", True)],
         ("model", "someone"): [("someone/a-model", True)]}
TREES = {"someone/sar-transfer-snow": [("README.md", 1200), ("train_images.npy", 20 * 10 ** 9),
                                       ("sub",), ("sub/val_images.npy", 5 * 10 ** 9)],
         "someone/sar-transfer-forty-v1": [("train_images.npy", 6 * 10 ** 9)],
         "someone/a-model": [("model.safetensors", 400 * 10 ** 6)],
         "someone/public-thing": [("big.bin", 900 * 10 ** 9)]}


def api_with(answers=None, repos=REPOS, trees=TREES, fail=None):
    return lambda token: FakeApi(token, answers or {READ: USER, WRITE: WRITER}, repos, trees, fail)


def run_check(read=READ, write=None, drive=None, **kw) -> tuple[dict, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        report = AC.check(read, write, drive_folder=drive, api=kw.pop("api", api_with()), **kw)
    text = out.getvalue()
    for token in (READ, WRITE):
        assert token not in text, "a token was printed"
    assert READ not in repr(report) and WRITE not in repr(report), "a token is in the report"
    return report, text


# ---- 1. the pieces ---------------------------------------------------------------------------
assert AC.GATED_REPO == DS.HF_ID and AC.HF == M.HF
assert AC.account_type({"isPro": True}) == "pro" and AC.account_type({"isPro": False}) == "free"
assert AC.account_type({}) == "unknown"
assert AC.write_rights(WRITER) == ("yes", "role write") and AC.write_rights(USER) == ("no", "role read")
fine = {"name": "someone", "auth": {"accessToken": {"role": "fineGrained", "fineGrained": {
    "canReadGatedRepos": True, "global": [],
    "scoped": [{"entity": {"type": "user", "name": "someone"}, "permissions": ["repo.content.read", "repo.write"]}]}}}}
assert AC.write_rights(fine)[0] == "yes"
fine["auth"]["accessToken"]["fineGrained"]["scoped"][0]["entity"]["name"] = "some-org"
assert AC.write_rights(fine)[0] == "no", "repo.write scoped to another entity counts as write"
fine["auth"]["accessToken"]["fineGrained"]["global"] = ["repo.write"]
assert AC.write_rights(fine)[0] == "yes"
assert AC.write_rights({"auth": {"accessToken": {"role": "odd"}}}) == ("unknown", "the answer holds the role 'odd'")
assert AC.write_rights({})[0] == "unknown" and AC.write_rights({"auth": None})[0] == "unknown"
low, high = AC.pilot_need_gb()
aerial_bytes = ((sum(A.SET_SIZES.values()) - A.SET_SIZES["distil_heldout"]) * 4
                + sum(A.SET_SIZES[s] for s in A.LABELLED)) * A.OUT_PX ** 2
assert abs(low - (D.N_PAIRS * 2.38e6 + aerial_bytes) / GB) < 1e-9
assert abs(high - (D.N_PILOT * 2.38e6 + aerial_bytes) / GB) < 1e-9
assert 55 < low < 57 and 69 < high < 71, (low, high)             # about 56 to 70 GB, as says
assert AC.verdict(31.0, 100.0, (56.0, 70.0)).startswith("does not fit, about 1 GB over")
assert AC.verdict(20.0, 100.0, (56.0, 70.0)).startswith("fits,")
assert AC.verdict(31.0, 1000.0, (56.0, 70.0)).startswith("fits,")
assert AC.hide("token READ x", ("READ", None)) == "token <token> x"
print("pieces verified, roles of both shapes, the need from the constants, the verdict")

# ---- 2. the HEAD on the gated file, the token only to huggingface.co -------------------------
AC.HF = BASE                                     # the local server stands for huggingface.co
try:
    for code, want in ((200, "yes"), (302, "yes"), (401, "no"), (403, "no"), (500, "unclear")):
        CODE["head"] = code
        HITS.clear()
        got, detail = AC.gated_read(READ)
        assert got == want and f"HTTP {code}" in detail, (code, got, detail)
        assert HITS[0][0] == f"/{AC.GATED_REPO}/resolve/main/config.json"
        assert HITS[0][1].get("Authorization") == f"Bearer {READ}", "the token did not reach the Hub"
        assert READ not in detail
    CODE["head"] = 200
    HITS.clear()
    got, _ = AC.gated_read(None)
    assert got == "yes" and "Authorization" not in HITS[0][1]
finally:
    AC.HF = M.HF
HITS.clear()
got, _ = AC.gated_read(READ, url=f"{BASE}/{AC.GATED_REPO}/resolve/main/config.json")   # not huggingface.co
assert got == "yes" and "Authorization" not in HITS[0][1], "the token went to another host"
got, detail = AC.gated_read(READ, url="http://127.0.0.1:1/x")                          # nothing listens
assert got == "unclear" and detail.startswith("no answer, ") and READ not in detail
print("HEAD verified, yes on 200 and 302, no on 401 and 403, unclear otherwise, token only to the Hub")

# ---- 3. the whole check -----------------------------------------------------------------------
AC.HF = BASE
try:
    CODE["head"] = 200
    drive = TMP / "drive"
    drive.mkdir(exist_ok=True)
    report, text = run_check(drive=drive)                              # read token only
    assert report["user"] == "someone" and report["account_type"] == "free" and report["gated_read"] == "yes"
    assert "user            someone (user), account type free" in text
    assert "gated DINOv3    yes, HTTP 200 on facebook/dinov3-vitl16-pretrain-sat493m/config.json" in text
    assert report["write_token"] == "missing" and "The upload cell of notebook 01 needs it" in text
    assert [r["repo"] for r in report["repos"]] == ["someone/sar-transfer-snow", "someone/sar-transfer-forty-v1",
                                                    "someone/a-model"], report["repos"]
    assert report["repos"][0] == {"repo": "someone/sar-transfer-snow", "type": "dataset", "files": 3,
                                  "bytes": 25 * 10 ** 9 + 1200}
    assert abs(report["used_gb"] - 31.4000012) < 1e-6 and report["quota_gb"] == 100.0
    assert "public-thing" not in text and "total 31.40 GB in 3 private repos" in text
    assert "private storage of a free account 100 GB (https://huggingface.co/docs/hub/storage-limits, table Storage plans, read 2026-09-30" in text
    assert report["verdict"].startswith("does not fit, about 1 GB over") and "verdict         does not fit" in text
    assert f"the pilot needs about {low:.0f} to {high:.0f} GB, pairs x 2.38 MB plus the aerial sets" in text
    assert report["drive_free_gb"] > 0 and "Drive  free " in text and str(drive) in text
    assert report["need_gb"] == (low, high)

    report, text = run_check(write=WRITE)                              # both tokens, PRO
    pro = {READ: {**USER, "isPro": True}, WRITE: WRITER}
    report, text = run_check(write=WRITE, api=api_with(answers=pro))
    assert report["account_type"] == "pro" and report["quota_gb"] == 1000.0
    assert report["write_token"] == "given" and report["write_rights"] == "yes"
    assert "user            someone, write rights yes (role write)" in text
    assert report["verdict"].startswith("fits,") and "private storage of a pro account 1,000 GB" in text

    CODE["head"] = 403                                                 # a token without access
    report, text = run_check()
    assert report["gated_read"] == "no" and "gated DINOv3    no, HTTP 403, the token has no access" in text
    CODE["head"] = 200

    report, text = run_check(write=WRITE, api=api_with(fail="whoami"))  # the Hub does not answer
    assert report["user"] is None and text.count("user            could not be read, ConnectionError, stand-in") == 2
    assert "<token>" in text and "not listed, the user is not known" in text
    assert report["verdict"] == "unknown, the private repos could not be listed" and report["repos"] == []
    assert "the account type is unknown so the free quota is taken" in text
    report, text = run_check(api=api_with(fail="list"))                # the listing fails
    assert report["user"] == "someone" and "could not be listed, OSError, stand-in" in text
    assert report["verdict"].startswith("unknown") and "<token>" in text

    heavy = {**TREES, "someone/a-model": [("model.safetensors", 60 * 10 ** 9)]}   # over the quota
    report, text = run_check(api=api_with(trees=heavy))
    assert abs(report["used_gb"] - 91.0000012) < 1e-6
    assert report["verdict"].startswith("does not fit, about 61 GB over"), report["verdict"]

    unknown = {READ: {k: v for k, v in USER.items() if k != "isPro"}}   # no isPro entry
    report, text = run_check(api=api_with(answers=unknown))
    assert report["account_type"] == "unknown" and report["quota_gb"] == 100.0
    assert "account type unknown" in text and "the free quota is taken" in text

    said = []
    report, text = run_check(say=said.append)                          # another printer
    assert text == "" and any(line.startswith("read token") for line in said)
finally:
    AC.HF = M.HF
    server.shutdown()
    socket.socket.connect = REAL_CONNECT
assert not NETWORK, f"the test asked for the network, {NETWORK}"
print("check verified, read token only, both tokens, no access, no answer, over the quota, unknown type")

shutil.rmtree(TMP, ignore_errors=True)
print("test_account: all checks passed")
