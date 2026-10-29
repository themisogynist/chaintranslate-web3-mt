"""Build English/French pairs by identical ethereum.org file and localization key."""

import argparse, concurrent.futures, hashlib, html, json, random, re, urllib.request
from pathlib import Path

REVISION = "fb3514bd596848d74500429c43b955c999af839b"
REPO = "ethereum/ethereum-org-website"


def fetch(url):
    with urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "ChainTranslate academic corpus preparation"}
        ),
        timeout=45,
    ) as r:
        return r.read()


def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    listing = json.loads(
        fetch(
            f"https://api.github.com/repos/{REPO}/contents/src/intl/en?ref={REVISION}"
        )
    )
    frlisting = json.loads(
        fetch(
            f"https://api.github.com/repos/{REPO}/contents/src/intl/fr?ref={REVISION}"
        )
    )
    common = sorted(
        {x["name"] for x in listing if x["name"].endswith(".json")}
        & {x["name"] for x in frlisting}
    )

    def one(name):
        docs = []
        for lang in ["en", "fr"]:
            url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/src/intl/{lang}/{name}"
            raw = fetch(url)
            docs.append(json.loads(raw))
        return name, docs

    pairs = []
    seen = set()
    discarded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name, (en, fr) in pool.map(one, common):
            for key in sorted(en.keys() & fr.keys()):
                if not isinstance(en[key], str) or not isinstance(fr[key], str):
                    continue
                a, b = clean(en[key]), clean(fr[key])
                if any(c in a + b for c in ["{", "}"]) or "http" in a + b:
                    continue
                if not 4 <= len(a.split()) <= 18 or not 3 <= len(b.split()) <= 30:
                    continue
                norm = a.casefold()
                if norm == b.casefold() or norm in seen:
                    discarded += 1
                    continue
                seen.add(norm)
                pairs.append(
                    dict(
                        id=hashlib.sha256((name + "::" + key).encode()).hexdigest()[
                            :16
                        ],
                        file=name,
                        key=key,
                        source=a,
                        reference=b,
                        source_url=f"https://github.com/{REPO}/blob/{REVISION}/src/intl/en/{name}",
                        reference_url=f"https://github.com/{REPO}/blob/{REVISION}/src/intl/fr/{name}",
                    )
                )
    groups = sorted({p["file"] for p in pairs})
    random.Random(42).shuffle(groups)
    n = len(groups)
    assign = {
        g: (
            "train"
            if i < int(0.8 * n)
            else "validation"
            if i < int(0.9 * n)
            else "test"
        )
        for i, g in enumerate(groups)
    }
    info = {}
    for split, limit in [("train", 512), ("validation", 64), ("test", 32)]:
        rows = [p for p in pairs if assign[p["file"]] == split]
        random.Random(42).shuffle(rows)
        available = len(rows)
        rows = rows[:limit]
        if len(rows) < limit:
            raise ValueError(
                f"Only {len(rows)} pairs for {split}; inspect corpus before changing splits"
            )
        (out / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2)
        )
        info[split] = dict(
            pairs=len(rows),
            available_pairs=available,
            files=len({p["file"] for p in rows}),
        )
    summary = dict(
        source="https://github.com/" + REPO,
        revision=REVISION,
        source_language="English",
        target_language="French",
        alignment="exact shared localization file and key",
        candidate_pairs=len(pairs),
        filtered_pairs_before_deduplication=len(pairs) + discarded,
        unique_pairs_after_deduplication=len(pairs),
        matched_json_files=len(common),
        duplicate_or_identical_pairs_removed=discarded,
        seed=42,
        splits=info,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    a = p.parse_args()
    prepare(a.out)
