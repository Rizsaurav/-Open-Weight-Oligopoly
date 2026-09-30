"""01_collect.py — Collect the Hugging Face Hub model listing (public API, no auth).

Endpoint: GET https://huggingface.co/api/models?limit=100&sort=downloads&direction=-1
Pagination: cursor via the `Link: <url>; rel="next"` response header.
Rate limit: 500 requests / 300 s (fixed window). We sleep 0.65 s between
requests (~460/window) and back off 300 s on HTTP 429.

The `downloads` field is the trailing 30-day download count (distinct from
`downloadsAllTime`, available only via expand=downloadsAllTime). We sort and
collect by this field, so the sample is the top-N models by recent downloads.

Assertion gates: every page must be a JSON list; every item must carry an
`id` and a non-negative integer `downloads`; final frame must be duplicate-
free and sorted non-increasing in downloads (the API's stated contract).

Usage: python3 01_collect.py [--target 20000] [--limit 100]
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone

BASE = "https://huggingface.co/api/models"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW = os.path.join(DATA, "raw_pages")
SLEEP = 0.65  # seconds between requests: ~460 requests per 300 s window

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hf-oligopoly-research/1.0 (academic audit)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8"), resp.headers.get("Link")
    except urllib.error.HTTPError as e:
        return e.code, "", ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)

    url = f"{BASE}?limit={args.limit}&sort=downloads&direction=-1"
    seen, pages, n_items = set(), 0, 0
    t0 = time.time()
    while n_items < args.target:
        status, body, link = fetch(url)
        if status == 429:
            print("HTTP 429: backing off 300 s", flush=True)
            time.sleep(300)
            continue
        assert status == 200, f"collection failed: HTTP {status} at page {pages}"
        try:
            items = json.loads(body)
        except json.JSONDecodeError:
            raise AssertionError(f"page {pages}: response is not valid JSON")
        assert isinstance(items, list), f"page {pages}: expected a JSON list"
        assert len(items) > 0, f"page {pages}: empty page before target reached"
        for m in items:
            assert isinstance(m, dict) and "id" in m, f"page {pages}: item missing 'id'"
            d = m.get("downloads")
            assert isinstance(d, int) and d >= 0, f"page {pages}: bad downloads for {m.get('id')}"
        with open(os.path.join(RAW, f"page_{pages:04d}.json"), "w") as f:
            json.dump(items, f)
        for m in items:
            if m["id"] not in seen:
                seen.add(m["id"])
                n_items += 1
        pages += 1
        m = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        assert m, f"page {pages}: no next-cursor Link header"
        url = m.group(1)
        if pages % 20 == 0:
            print(f"  ... {n_items} models, {pages} pages, {time.time()-t0:.0f}s", flush=True)
        time.sleep(SLEEP)

    # ---- consolidate ----
    recs = []
    for p in range(pages):
        with open(os.path.join(RAW, f"page_{p:04d}.json")) as f:
            for m in json.load(f):
                if m["id"] not in [r["id"] for r in recs]:
                    recs.append(m)
    # dedupe preserving order (order-preserving; the loop above already deduped,
    # this is a defensive second pass)
    seen_ids, deduped = set(), []
    for r in recs:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            deduped.append(r)
    recs = deduped
    recs.sort(key=lambda r: -r["downloads"])
    recs = recs[:args.target]
    with open(os.path.join(DATA, "models_list.json"), "w") as f:
        json.dump({"collected_at": datetime.now(timezone.utc).isoformat(),
                   "n": len(recs), "target": args.target, "models": recs}, f)

    # ---- final gates ----
    assert len(recs) >= int(0.95 * args.target), \
        f"collected only {len(recs)} of {args.target} target"
    ids = [r["id"] for r in recs]
    assert len(set(ids)) == len(ids), "duplicate model ids in final frame"
    dl = [r["downloads"] for r in recs]
    assert all(b <= a for a, b in zip(dl, dl[1:])), "downloads not non-increasing: API sort contract broken"
    assert all(isinstance(x, int) and x >= 0 for x in dl), "bad downloads values"
    print(f"OK: {len(recs)} models, {pages} pages, {time.time()-t0:.0f}s")
    print(f"top: {recs[0]['id']} ({recs[0]['downloads']:,}) | "
          f"bottom: {recs[-1]['id']} ({recs[-1]['downloads']:,})")

if __name__ == "__main__":
    main()
