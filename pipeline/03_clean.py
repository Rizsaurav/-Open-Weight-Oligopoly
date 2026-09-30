"""03_clean.py — Parse list + detail API payloads into an analysis-ready frame.

Inputs:  data/models_list.json, data/raw_detail/*.json
Outputs: data/models_clean.csv, data/cleaning_report.json

Feature engineering (all rules explicit and documented):
- org: id prefix before '/', or the id itself for legacy single-component ids.
- org_type (author-coded, case-insensitive curated lists; everything else
  -> 'community'):
    corporate: for-profit companies and their labs.
    research:  academic / nonprofit / public research labs.
    community: individuals and community converter/quant groups.
- license_class from `license:*` tags, first match wins in this order:
    restrictive: openrail*, *llama*, gemma, *nc* (non-commercial), apple-amlr,
                 openmdw-1.1, bigscience-bloom-rail*
    copyleft:    gpl*, agpl*, lgpl*, cc-by-sa*
    permissive:  apache-2.0, mit, bsd-*, cc-by-*, cc0-1.0, wtfpl, unlicense,
                 isc, cdla-permissive-2.0, afl-3.0
    other:       license:other / license:unknown / any other license tag
    unspecified: no license:* tag at all
- age_days vs the collection timestamp; year / quarter from createdAt.
- params from detail safetensors.total; has_params flag.

Assertion gates: row count matches the manifest; ids unique; downloads are
non-negative ints; every createdAt parses with year in [2015, 2026];
age_days >= 0; params positive where present; license/org_type take only
declared values. Coverage of pipeline_tag, license tags, and params is
reported (not imputed away).
"""
import json, os, glob, re
from datetime import datetime, timezone
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

CORPORATE = {
    "qwen", "alibaba", "alibaba-nlp", "google", "google-bert", "google-t5",
    "facebook", "facebookai", "meta", "meta-llama", "microsoft", "openai",
    "openai-community", "nvidia", "amazon", "apple", "deepseek-ai", "baidu",
    "tencent", "bytedance", "mistralai", "anthropic", "cohere", "ai21labs",
    "xai", "ibm", "ibm-granite", "intel", "redhatai", "stabilityai", "samsung",
    "huawei", "tiiuae", "adept", "nomic-ai", "voyageai", "jinaai", "upstage",
    "lgai", "naver", "kakao", "sony", "oracle", "dell", "hpe", "cisco",
    "sap", "adobe", "salesforce", "snowflake", "databricks", "anyscale",
    "together", "fireworks", "groq", "cerebras", "sambanova", "dmeta",
    "zyphra", "recraft", "black-forest-labs", "ideogram", "luma", "pika",
    "runway", "elevenlabs", "suno", "udio", "perplexity", "youdao",
}
RESEARCH = {
    "sentence-transformers", "cross-encoder", "allenai", "helsinki-nlp",
    "baai", "bigscience", "eleutherai", "laion", "carperai", "bigcode",
    "nyu", "stanford", "berkeley", "mit", "cmu", "uw", "princeton",
    "cornell", "oxford", "cambridge", "eth", "epfl", "inria", "mpi",
    "vectorinstitute", "mila", "turing", "australian", "monash",
}

def license_class(tags):
    lic = [t[len("license:"):] for t in tags if t.startswith("license:")]
    if not lic:
        return "unspecified"
    blob = " ".join(lic)
    if re.search(r"openrail|llama|gemma|nc[\W_]|noncommercial|apple-amlr|openmdw|bloom-rail", blob):
        return "restrictive"
    if re.search(r"\bgpl|\bagpl|\blgpl|cc-by-sa", blob):
        return "copyleft"
    if re.search(r"apache-2\.0|^mit$|bsd-|cc-by-[\d.]|cc0-1\.0|wtfpl|unlicense|\bisc$|cdla-permissive|afl-3\.0", blob):
        return "permissive"
    return "other"

def main():
    man = json.load(open(os.path.join(DATA, "models_list.json")))
    models = man["models"]
    collected_at = datetime.fromisoformat(man["collected_at"])
    assert len(models) == man["n"], "manifest count mismatch"

    # detail -> params lookup
    params, detail_ok = {}, 0
    for p in glob.glob(os.path.join(DATA, "raw_detail", "*.json")):
        rec = json.load(open(p))
        mid = rec["requested_id"]
        st = (rec["detail"].get("safetensors") or {}).get("total")
        params[mid] = st if isinstance(st, int) and st > 0 else None
        detail_ok += 1

    rows = []
    for m in models:
        oid = m["id"]
        org = oid.split("/")[0] if "/" in oid else oid
        ol = org.lower()
        otype = "corporate" if ol in CORPORATE else ("research" if ol in RESEARCH else "community")
        try:
            ca = datetime.fromisoformat(m["createdAt"].replace("Z", "+00:00"))
        except Exception:
            raise AssertionError(f"unparseable createdAt for {oid}: {m.get('createdAt')}")
        rows.append({
            "id": oid, "org": org, "org_type": otype,
            "has_org_prefix": int("/" in oid),
            "downloads": m["downloads"], "likes": m.get("likes") or 0,
            "pipeline_tag": m.get("pipeline_tag"),
            "license_class": license_class(m.get("tags") or []),
            "created_at": ca.isoformat(),
            "year": ca.year,
            "quarter": f"{ca.year}-Q{(ca.month - 1) // 3 + 1}",
            "age_days": (collected_at - ca).days,
            "params": params.get(oid),
        })
    df = pd.DataFrame(rows)
    df["has_params"] = df.params.notna().astype(int)

    # ---------------- gates ----------------
    assert len(df) == len(models), "row loss in cleaning"
    assert df.id.nunique() == len(df), "duplicate ids after cleaning"
    assert (df.downloads >= 0).all() and pd.api.types.is_integer_dtype(df.downloads), "bad downloads"
    assert df.year.between(2015, 2026).all(), f"createdAt year out of range: {sorted(df.year.unique())}"
    assert (df.age_days >= 0).all(), "negative age_days"
    assert (df.org.str.len() > 0).all(), "empty org"
    assert set(df.org_type.unique()) <= {"corporate", "research", "community"}, "bad org_type"
    assert set(df.license_class.unique()) <= {"permissive", "copyleft", "restrictive", "other", "unspecified"}, \
        f"bad license_class: {set(df.license_class.unique())}"
    pp = df.loc[df.has_params == 1, "params"]
    assert (pp > 0).all(), "non-positive params"
    assert (df.likes >= 0).all(), "negative likes"

    df.to_csv(os.path.join(DATA, "models_clean.csv"), index=False)
    report = {
        "n": int(len(df)), "n_orgs": int(df.org.nunique()),
        "collected_at": man["collected_at"],
        "coverage": {
            "pipeline_tag": round(float(df.pipeline_tag.notna().mean()), 4),
            "license_tagged": round(float((df.license_class != "unspecified").mean()), 4),
            "params": round(float(df.has_params.mean()), 4),
            "detail_fetched": detail_ok,
        },
        "org_type_counts": df.org_type.value_counts().to_dict(),
        "license_class_counts": df.license_class.value_counts().to_dict(),
        "year_range": [int(df.year.min()), int(df.year.max())],
    }
    json.dump(report, open(os.path.join(DATA, "cleaning_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    print("\ncleaning OK:", os.path.join(DATA, "models_clean.csv"))

if __name__ == "__main__":
    main()
