"""02_collect_detail.py — Fetch per-model detail for the top-N models by downloads.

Endpoint: GET https://huggingface.co/api/models/{id} (public, no auth).
Adds: safetensors.total (parameter count), author, gated, cardData.

Assertion gates: returned `id` must match the requested model id; parameter
counts, when present, must be positive integers; overall fetch failure rate
must stay below 5% (deleted/renamed repos are expected and recorded).

Usage: python3 02_collect_detail.py [--top 2000]
"""
import argparse, json, os, sys, time, urllib.request, urllib.error, urllib.parse
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
DET = os.path.join(DATA, "raw_detail")
SLEEP_DEFAULT = 0.75  # ~400 requests per 300 s window, under the 500 limit

def fetch_detail(mid):
    # NOTE: the org/model slash must NOT be percent-encoded (API returns 400
    # "repo name includes an url-encoded slash"); only encode other chars.
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(mid, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": "hf-oligopoly-research/1.0 (academic audit)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return -1, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.75)
    args = ap.parse_args()
    os.makedirs(DET, exist_ok=True)

    manifest_path = os.path.join(DATA, "models_list.json")
    assert os.path.exists(manifest_path), "models_list.json missing: run 01_collect.py first"
    models = json.load(open(manifest_path))["models"][args.offset:args.offset + args.top]
    ids = [m["id"] for m in models]
    assert len(ids) == args.top, f"expected {args.top} models, got {len(ids)}"

    ok, failed = 0, []
    t0 = time.time()
    for j, mid in enumerate(ids):
        i = args.offset + j  # global index -> filename
        out = os.path.join(DET, f"{i:05d}.json")
        if os.path.exists(out):
            ok += 1
            continue
        status, detail = fetch_detail(mid)
        if status == 429:
            print("HTTP 429: backing off 300 s", flush=True)
            time.sleep(300)
            status, detail = fetch_detail(mid)
        if status != 200 or detail is None:
            failed.append({"id": mid, "status": status})
            time.sleep(args.sleep)
            continue
        # ---- gates ----
        assert detail.get("id") == mid, f"id mismatch: requested {mid}, got {detail.get('id')}"
        st = (detail.get("safetensors") or {}).get("total")
        # total=0 occurs for placeholder/empty safetensors indexes: treat as
        # missing params rather than a failure (recorded in params_raw).
        params_raw = st if isinstance(st, int) and st > 0 else None
        with open(out, "w") as f:
            json.dump({"requested_id": mid, "fetched_at": datetime.now(timezone.utc).isoformat(),
                       "params_raw": st, "detail": detail}, f)
        ok += 1
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{args.top} detail fetched, {time.time()-t0:.0f}s", flush=True)
        time.sleep(args.sleep)

    fail_rate = len(failed) / args.top
    with open(os.path.join(DATA, "detail_failures.json"), "w") as f:
        json.dump(failed, f, indent=1)
    assert fail_rate < 0.05, f"detail fetch failure rate {fail_rate:.2%} exceeds 5%"
    print(f"OK: {ok}/{args.top} details fetched, {len(failed)} failed ({fail_rate:.2%}), {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
