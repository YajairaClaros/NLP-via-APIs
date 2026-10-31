"""Local replica of the instructor's evaluation harness.

Calls a running service exactly like the live evaluation does (one item at a time, 5 s
timeout per request, 150 s global cap), checks the contract and access control, scores
the answers and projects the "live evaluation" points with the grading formula.

    export QA_API_TOKEN=...            # PowerShell: $env:QA_API_TOKEN = "..."
    python -m eval.harness --url http://localhost:8000
    python -m eval.harness --url https://<name>.ngrok-free.app --data eval/shadow_set.json
    python -m eval.harness --url http://localhost:8000 --data eval/challenge_set.json   # decoder investigation

Item score (Section XIV.1): 1 for exact match or a correct null; partial overlaps are judged
by an LLM in the real evaluation (1 / 0.5 / 0) -- here we report them separately and use
0.5 as an estimate; 0 for wrong null, answering an unanswerable, timeout or invalid response.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from qa.metrics import score_item

ROOT = Path(__file__).resolve().parents[1]
BASELINE, REFERENCE = 0.25, 0.70     # provisional anchors from the brief
PER_REQUEST_TIMEOUT, GLOBAL_CAP = 5.0, 150.0


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, ok: bool, label: str, extra: str = ""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + extra) if extra and not ok else ''}")
        self.failed += not ok
        return ok


def contract_checks(url: str, token: str, c: Checks):
    print("\n== Health and access control")
    r = requests.get(f"{url}/health", timeout=PER_REQUEST_TIMEOUT)
    c.check(r.status_code == 200 and r.json().get("status") == "ok", "GET /health -> 200 ok (public)", r.text[:100])

    for label, headers in [("no token", {}), ("wrong token", {"Authorization": "Bearer wrong"}),
                           ("lowercase scheme", {"Authorization": f"bearer {token}"}),
                           ("token without scheme", {"Authorization": token})]:
        r = requests.get(f"{url}/info", headers=headers, timeout=PER_REQUEST_TIMEOUT)
        c.check(r.status_code == 401 and r.json().get("error") == "unauthorized", f"GET /info {label} -> 401", f"{r.status_code} {r.text[:80]}")
    r = requests.post(f"{url}/predict", json={"id": "x", "context": "a", "question": "b"}, timeout=PER_REQUEST_TIMEOUT)
    c.check(r.status_code == 401, "POST /predict no token -> 401", str(r.status_code))

    auth = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{url}/info", headers=auth, timeout=PER_REQUEST_TIMEOUT)
    info = r.json() if r.ok else {}
    ok = (r.status_code == 200 and isinstance(info.get("team"), str) and " " not in info.get("team", " ")
          and isinstance(info.get("model"), str) and isinstance(info.get("params"), int))
    c.check(ok, "GET /info valid token -> 200 with team, model, params", r.text[:120])
    if info:
        print(f"         {info}")
        if info.get("team") == "team-XX":
            print("         WARNING: team is still 'team-XX'. Set QA_TEAM to your registered team id.")

    print("\n== Request validation")
    bad = [("missing question", {"id": "x", "context": "abc"}),
           ("context > 5000 chars", {"id": "x", "context": "a" * 5001, "question": "q"}),
           ("extra field", {"id": "x", "context": "abc", "question": "q", "foo": 1})]
    for label, payload in bad:
        r = requests.post(f"{url}/predict", headers=auth, json=payload, timeout=PER_REQUEST_TIMEOUT)
        c.check(r.status_code in (400, 422) and r.json().get("error") == "bad_request", f"{label} -> 400/422 bad_request", f"{r.status_code} {r.text[:80]}")
    return auth


def validate(resp: requests.Response, item: dict) -> tuple[dict | None, str | None]:
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    try:
        d = resp.json()
    except ValueError:
        return None, "malformed JSON"
    for k in ("id", "answer", "start", "score"):
        if k not in d:
            return None, f"missing field {k!r}"
    if d["id"] != item["id"]:
        return None, "id mismatch"
    a, s = d["answer"], d["start"]
    if (a is None) != (s is None):
        return None, "answer/start null mismatch"
    if a is not None and (not isinstance(s, int) or item["context"][s:s + len(a)] != a):
        return None, "context[start:start+len(answer)] != answer"
    return d, None


def run_items(url: str, auth: dict, items: list[dict]) -> list[dict]:
    rows, t_global = [], time.perf_counter()
    for it in items:
        elapsed = time.perf_counter() - t_global
        if elapsed > GLOBAL_CAP:
            rows.append({"id": it["id"], "status": "global cap exceeded", "item_score": 0.0})
            continue
        payload = {"id": it["id"], "context": it["context"], "question": it["question"]}
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{url}/predict", headers=auth, json=payload, timeout=PER_REQUEST_TIMEOUT)
            d, err = validate(r, it)
        except requests.Timeout:
            d, err = None, "timeout (> 5 s)"
        except requests.RequestException as e:
            d, err = None, f"connection error: {e.__class__.__name__}"
        ms = (time.perf_counter() - t0) * 1000
        row = {"id": it["id"], "question": it["question"], "latency_ms": round(ms),
               "server_ms": r.headers.get("X-Inference-Ms") if d else None}

        if "expected_free_text" in it:          # challenge items: not scorable as spans
            row.update(status=err or "ok", answer=d and d["answer"], expected=it["expected_free_text"],
                       category=it.get("category"), item_score=None)
        elif err:
            row.update(status=err, item_score=0.0)
        else:
            golds = [a["text"] for a in it.get("answers", [])]
            em, f1 = score_item(d["answer"], golds)
            if em == 1.0:
                status, s = ("correct null" if not golds else "exact match"), 1.0
            elif not golds:
                status, s = "answered an unanswerable", 0.0
            elif d["answer"] is None:
                status, s = "wrong null", 0.0
            elif f1 > 0:
                status, s = "partial overlap (LLM-judged live; estimated 0.5)", 0.5
            else:
                status, s = "wrong span", 0.0
            row.update(status=status, answer=d["answer"], start=d["start"], score=d["score"],
                       gold=golds, em=em, f1=round(f1, 3), item_score=s)
        rows.append(row)
        mark = {1.0: "OK ", 0.5: "~  ", 0.0: "X  ", None: "?  "}[row["item_score"]]
        print(f"  {mark}{it['id']:<14} {row['latency_ms']:>5} ms  {row['status']:<26} -> {row.get('answer')!r}"[:150])
    return rows


def summarize(rows: list[dict], total_s: float):
    scored = [r for r in rows if r["item_score"] is not None]
    if not scored:
        return {}
    n = len(scored)
    est = sum(r["item_score"] for r in scored) / n
    strict = sum(r["item_score"] == 1.0 for r in scored) / n
    optimistic = sum(1.0 if r["item_score"] > 0 else 0.0 for r in scored) / n
    lat = sorted(r["latency_ms"] for r in rows)
    points = lambda x: max(0.0, min(1.0, (x - BASELINE) / (REFERENCE - BASELINE))) * 60
    by_status = {}
    for r in scored:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "items": n, "total_seconds": round(total_s, 1),
        "projected_50_items_seconds": round(total_s / max(len(rows), 1) * 50, 1),
        "latency_ms_mean": round(sum(lat) / len(lat)), "latency_ms_max": lat[-1],
        "team_score_estimate": round(est, 3), "team_score_strict": round(strict, 3),
        "team_score_optimistic": round(optimistic, 3),
        "live_points_estimate": round(points(est), 1), "live_points_strict": round(points(strict), 1),
        "by_status": by_status,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--data", default=str(ROOT / "docs" / "brief" / "public_sample.json"))
    p.add_argument("--token", default=os.environ.get("QA_API_TOKEN"), help="default: $QA_API_TOKEN")
    p.add_argument("--skip_contract", action="store_true")
    p.add_argument("--out", default=None, help="results json (default: eval/results/<data>_<time>.json)")
    a = p.parse_args()
    if not a.token:
        sys.exit("Set QA_API_TOKEN or pass --token")
    url = a.url.rstrip("/")
    items = json.loads(Path(a.data).read_text(encoding="utf-8"))["items"]

    c = Checks()
    auth = {"Authorization": f"Bearer {a.token}"}
    if not a.skip_contract:
        auth = contract_checks(url, a.token, c)

    print(f"\n== Items: {Path(a.data).name} ({len(items)})")
    t0 = time.perf_counter()
    rows = run_items(url, auth, items)
    summary = summarize(rows, time.perf_counter() - t0)

    print("\n== Summary")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    if summary.get("projected_50_items_seconds", 0) > GLOBAL_CAP * 0.7:
        print("  WARNING: projected time for 50 items is close to the 150 s global cap.")
    if c.failed:
        print(f"\n  {c.failed} contract/access-control check(s) FAILED.")

    out = Path(a.out) if a.out else ROOT / "eval" / "results" / f"{Path(a.data).stem}_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"url": url, "data": a.data, "summary": summary, "contract_failures": c.failed,
                               "items": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
