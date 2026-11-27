#!/usr/bin/env python3
"""A Flash-Next pack's drafter changes no token and no logit row (`--serve`, greedy).

    draft_identity_test.py ENGINE MODEL.basalt [--drafter NAME] [--tokens N] [--fixtures DIR] [ENGINE ARGS...]

Two engine sessions run the same requests: one with `--spec 0` (one-token windows, no drafter) and one with the pack's
drafter at the engine's default window. Drafts only choose which rows a window checks, so every request must give the
same tokens in both, and every logit row the plain session hashed (`--exp concur9_logit_hash=1`, `LH` lines on stderr)
must be among the drafter session's.

The requests cover what a drafter's context has to follow: a new conversation, the next turn of it (read on from the
live session), a turn that rewinds into it, a prompt read in chunks, a prompt longer than the drafter's context, and
the first conversation again after the others.

The experts' placement must be the same in both sessions, or rows round differently: pass a fixed tier in ENGINE ARGS
(`--expert-cache N`, and a fixed peer tier when there is one). Adaptive swaps are turned off here. With
`--conversation-cache-mib N` in ENGINE ARGS the last request restores a parked conversation instead of reading it again.

`--drafter NAME` also requires the drafter session to report `drafter=NAME` on its INFO line (`block`, `mtp`).
Exit 0: identical. Exit 1: a difference, or the drafter session offered no draft. Exit 2: an engine failed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINNED = ["--adapt-swaps", "0", "--peer-adapt-swaps", "0", "--exp", "concur9_logit_hash=1"]


class EngineFailed(RuntimeError):
    pass


class Engine:
    """One `--serve` session: its INFO facts, and requests run to their DONE."""

    def __init__(self, cmd: list[str], log_path: Path):
        self.log = open(log_path, "w")
        self.info: dict[str, str] = {}
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise EngineFailed(f"the engine exited before READY (see {log_path})")
            if line.startswith("INFO "):
                for word in line.split()[1:]:
                    key, sep, value = word.partition("=")
                    if sep:
                        self.info[key] = value
            if line.startswith("READY"):
                break

    def run(self, ids: list[int], max_new: int) -> tuple[list[int], int, int, int]:
        """One greedy GEN: its tokens, the prompt tokens it reused, and the drafts accepted and offered."""
        self.p.stdin.write(f"GEN {max_new} temperature=0 {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        tokens: list[int] = []
        reused = 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise EngineFailed("the engine exited inside a request")
            f = line.split()
            if not f:
                continue
            if f[0] == "ERR":
                raise EngineFailed(line.strip())
            if f[0] == "T":
                tokens.append(int(f[1]))
            elif f[0] == "RESUME":
                reused = int(f[1])
            elif f[0] == "DONE":
                # DONE <produced> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> ...
                return tokens, reused, int(f[6]), int(f[7])

    def stop(self) -> None:
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=120)
        except Exception:
            self.p.kill()
            self.p.wait()
        finally:
            self.log.close()


def fixture(path: Path, line: int = 0) -> list[int]:
    return [int(t) for t in path.read_text().splitlines()[line].split(",") if t.strip()]


def logit_hashes(log_path: Path) -> set[tuple[int, str]]:
    """The (position, hash) pairs of a session's `LH` lines."""
    pairs: set[tuple[int, str]] = set()
    for line in log_path.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) == 4 and f[0] == "LH":
            pairs.add((int(f[2]), f[3]))
    return pairs


def session(label: str, cmd: list[str], log_path: Path, conv: list[int], extra_turn: list[int], chunked: list[int],
            long_prompt: list[int], tokens: int) -> dict:
    """The requests, in order, on one engine: each one's tokens, and the session's facts."""
    eng = Engine(cmd, log_path)
    out: dict = {"info": eng.info, "tokens": {}, "reused": {}, "accepted": 0, "offered": 0}

    def request(name: str, ids: list[int], max_new: int) -> list[int]:
        got, reused, accepted, offered = eng.run(ids, max_new)
        out["tokens"][name] = got
        out["reused"][name] = reused
        out["accepted"] += accepted
        out["offered"] += offered
        print(f"{label}: {name}: {len(ids)} prompt tokens ({reused} reused), {len(got)} generated, "
              f"{accepted} of {offered} drafts accepted", flush=True)
        return got

    try:
        first = request("new", conv, tokens)
        turn = conv + first + extra_turn
        request("next turn", turn, tokens)
        request("rewound", conv[: max(2, 2 * len(conv) // 3)] + extra_turn, tokens)
        request("chunked", chunked, tokens)
        if long_prompt:
            request("long", long_prompt, min(tokens, 48))
        request("first again", turn, tokens)
    finally:
        eng.stop()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("model")
    ap.add_argument("--drafter", default="", help="the drafter the second session must report (block, mtp)")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--fixtures", default=str(ROOT / "data" / "bench"))
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if "--expert-cache" not in extra:
        print("draft_identity_test: pass a fixed tier (--expert-cache N): the two sessions hold different amounts of VRAM, "
              "and auto-sized tiers would place the experts differently", file=sys.stderr)
        return 2
    fixtures = Path(args.fixtures)
    conv, chunked, extra_turn = fixture(fixtures / "conv.ids"), fixture(fixtures / "struct.ids"), fixture(fixtures / "prose.ids")
    long_path = fixtures / "prefill.ids"
    long_prompt = max((fixture(long_path, i) for i in range(len(long_path.read_text().splitlines()))), key=len) \
        if long_path.is_file() else []
    context = max(8192, len(long_prompt) + 2048)   # room for the long prompt and its reply
    base = [args.engine, "--model", args.model, "--serve", "--max-context", str(context), *PINNED, *extra]

    results = {}
    with tempfile.TemporaryDirectory(prefix="basalt_draft_identity_") as tmp:
        try:
            for label, spec in (("plain", ["--spec", "0"]), ("drafter", [])):
                log_path = Path(tmp) / f"{label}.err"
                results[label] = session(label, base + spec, log_path, conv, extra_turn, chunked, long_prompt, args.tokens)
                results[label]["hashes"] = logit_hashes(log_path)
        except EngineFailed as e:
            print(f"draft_identity_test: {e}", file=sys.stderr)
            return 2

    plain, drafter = results["plain"], results["drafter"]
    ok = True
    for name, ref in plain["tokens"].items():
        new = drafter["tokens"].get(name, [])
        if ref != new:
            i = next((k for k, (x, y) in enumerate(zip(ref, new)) if x != y), min(len(ref), len(new)))
            print(f"FAIL {name}: token {i} differs: plain {ref[i:i + 5]} drafter {new[i:i + 5]}")
            ok = False
    for label, r in results.items():   # the same prompt gives the same reply whatever the session read before it
        if r["tokens"].get("next turn") != r["tokens"].get("first again"):
            print(f"FAIL {label}: the next turn's reply changed when it was read again")
            ok = False
    # every row the plain session hashed is a row of the drafter session (which also hashes the prompt rows its first
    # window folds in)
    missing = plain["hashes"] - drafter["hashes"]
    if not plain["hashes"]:
        print("FAIL the plain session printed no logit hashes")
        ok = False
    elif missing:
        pos, h = min(missing)
        print(f"FAIL {len(missing)} logit rows of the plain session are not the drafter session's "
              f"(the first: position {pos}, hash {h})")
        ok = False
    if plain["offered"] != 0:
        print(f"FAIL the plain session offered {plain['offered']} drafts")
        ok = False
    if drafter["offered"] <= 0:
        print("FAIL the drafter session offered no draft")
        ok = False
    name = drafter["info"].get("drafter", "")
    if args.drafter and name != args.drafter:
        print(f"FAIL the drafter session reports drafter={name or '(none)'}, expected {args.drafter}")
        ok = False
    print(f"drafter={name or '(not reported)'}: {drafter['accepted']} of {drafter['offered']} drafts accepted, "
          f"{len(plain['hashes'])} logit rows compared")
    print("draft_identity_test: PASS" if ok else "draft_identity_test: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
