"""Validate recorded runs and aggregate their measured outputs."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import metrics

MODELS = ["basic", "mt5", "llama", "gemma"]
TEMPERATURES = [0.5, 1.0, 1.5]


def validate_run(model, score, translations, temperatures, test):
    """Reject missing, mixed or misaligned predictions before reporting scores."""
    expected = [r["id"] for r in test]
    if score["model"] != model or score["test_pairs"] != len(test):
        raise ValueError(f"{model}: score metadata does not match the test set")
    if [r["id"] for r in translations] != expected:
        raise ValueError(f"{model}: test IDs/order do not match")
    reference = {r["id"]: r for r in test}
    for row in translations + temperatures:
        original = reference[row["id"]]
        if row["model"] != model or any(
            row[k] != original[k] for k in ["source", "reference"]
        ):
            raise ValueError(f"{model}: mismatched source/reference for {row['id']}")
    actual = metrics(
        [r["prediction"] for r in translations], [r["reference"] for r in translations]
    )
    for name in ["bleu", "chrf"]:
        if not math.isclose(score[name], actual[name], abs_tol=1e-7):
            raise ValueError(f"{model}: {name} differs from the raw predictions")
    if len(temperatures) != 3 * min(4, len(test)):
        raise ValueError(f"{model}: missing or duplicate temperature samples")
    for temperature in TEMPERATURES:
        group = [r for r in temperatures if r["temperature"] == temperature]
        if [r["id"] for r in group] != expected[:4]:
            raise ValueError(f"{model}: temperature test IDs/order do not match")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(out="results", data="data", report="RESULTS.md"):
    out = Path(out)
    test = json.loads((Path(data) / "test.json").read_text())
    rows, translations, temperatures = [], [], []
    for model in MODELS:
        score_path = out / f"{model}_metrics.json"
        if not score_path.exists():
            continue
        score = json.loads(score_path.read_text())
        predictions = json.loads((out / f"{model}_translations.json").read_text())
        samples = json.loads((out / f"{model}_temperatures.json").read_text())
        validate_run(model, score, predictions, samples, test)
        rows.append(score)
        translations.extend(predictions)
        temperatures.extend(samples)
    missing = [m for m in MODELS if m not in {r["model"] for r in rows}]
    status = dict(
        complete=not missing,
        completed_models=[r["model"] for r in rows],
        missing_models=missing,
    )
    (out / "status.json").write_text(json.dumps(status, indent=2))
    if not rows:
        raise SystemExit("No completed experiments to report")
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "translations.csv", translations)
    lines = ["CHAINTRANSLATE MODEL OUTPUTS", ""]
    for r in translations:
        lines.extend(
            [
                f"MODEL: {r['model']} | ID: {r['id']}",
                "ENGLISH: " + r["source"],
                "FRENCH REFERENCE: " + r["reference"],
                "MODEL TRANSLATION: " + r["prediction"],
                "",
            ]
        )
    (out / "translations.txt").write_text("\n".join(lines), encoding="utf-8")
    summary = []
    for model in MODELS:
        for temperature in TEMPERATURES:
            group = [
                r
                for r in temperatures
                if r["model"] == model and r["temperature"] == temperature
            ]
            if not group:
                continue
            tokens = [r["prediction"].lower().split() for r in group]
            bigrams = [pair for words in tokens for pair in zip(words, words[1:])]
            counts, n = Counter(bigrams), len(bigrams)
            summary.append(
                dict(
                    model=model,
                    temperature=temperature,
                    pairs=len(group),
                    distinct_2=len(counts) / n if n else 0.0,
                    repeated_bigram_fraction=sum(c - 1 for c in counts.values()) / n
                    if n
                    else 0.0,
                    mean_words=sum(map(len, tokens)) / len(tokens),
                    **metrics(
                        [r["prediction"] for r in group],
                        [r["reference"] for r in group],
                    ),
                )
            )
    (out / "temperature_metrics.json").write_text(json.dumps(summary, indent=2))
    (out / "temperature_samples.json").write_text(
        json.dumps(temperatures, ensure_ascii=False, indent=2)
    )
    checks = []
    for row in translations:

        def numbers(s):
            # French decimal commas and English decimal points represent the same value.
            return {x.replace(",", ".") for x in re.findall(r"\b\d+(?:[.,]\d+)?\b", s)}

        missing_numbers = sorted(numbers(row["source"]) - numbers(row["prediction"]))
        added_numbers = sorted(numbers(row["prediction"]) - numbers(row["source"]))
        missing_terms = [
            term
            for term in ["Ethereum", "ETH", "Bitcoin", "USDC", "NFT"]
            if re.search(r"\b" + term + r"\b", row["source"])
            and not re.search(r"\b" + term + r"\b", row["prediction"], re.I)
        ]
        if (
            missing_numbers
            or added_numbers
            or missing_terms
            or not row["prediction"].strip()
        ):
            checks.append(
                dict(
                    model=row["model"],
                    id=row["id"],
                    missing_numbers=missing_numbers,
                    added_numbers=added_numbers,
                    missing_terms=missing_terms,
                    empty=not bool(row["prediction"].strip()),
                )
            )
    (out / "review_candidates.json").write_text(json.dumps(checks, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, key in zip(axes, ["bleu", "chrf"]):
        bars = ax.bar(
            [r["model"] for r in rows],
            [r[key] for r in rows],
            color=["#276678", "#52854c"][: len(rows)] if len(rows) <= 2 else "#276678",
        )
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_title(key.upper())
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (0–100)")
    fig.savefig(out / "scores.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for model in MODELS:
        history_path = out / f"{model}_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
            for ax, field in zip(axes, ["train_loss", "validation_loss"]):
                ax.plot(
                    [h["epoch"] for h in history],
                    [h[field] for h in history],
                    marker=".",
                    label=model,
                )
    for ax, title in zip(axes, ["Training loss", "Validation token loss"]):
        ax.set(title=title, xlabel="Epoch", ylabel="Cross entropy")
        ax.legend()
    fig.savefig(out / "learning_curves.png", dpi=180)
    plt.close(fig)
    text = [
        "# Experimental results",
        "",
        "Status: "
        + ("complete" if not missing else "incomplete; missing " + ", ".join(missing)),
        "",
        "| Model | BLEU | chrF | Adaptation seconds | Inference seconds | Base parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        train = (
            "N/A (prompting)"
            if row["training_seconds"] is None
            else f"{row['training_seconds']:.2f}"
        )
        text.append(
            f"| {row['model']} | {row['bleu']:.2f} | {row['chrf']:.2f} | {train} | {row['inference_seconds']:.2f} | {row['parameters']:,} |"
        )
    text += [
        "",
        "All reported runs are checked against the same test IDs, source strings and references; BLEU and chrF are recomputed from raw predictions. Scores are corpus-level SacreBLEU BLEU (13a) and chrF (character order 6, beta 2, no word n-grams). Signatures are retained in metrics.csv.",
        "",
        "Adaptation time includes training, validation and checkpoint writes, excluding downloads. Inference includes tokenization and decoding, excluding one warm-up call. Hardware is recorded per model. Runs on different hardware, or overlapping runs, do not establish an isolated speed comparison.",
        "",
        "## Sampling temperature",
        "",
        "| Model | Temperature | BLEU | chrF | Distinct 2 | Repeated bigrams |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        text.append(
            f"| {row['model']} | {row['temperature']:.1f} | {row['bleu']:.2f} | {row['chrf']:.2f} | {row['distinct_2']:.3f} | {row['repeated_bigram_fraction']:.3f} |"
        )
    text += [
        "",
        "Temperature metrics use the same four examples at each temperature and seed 42. Distinct-2 and repeated-bigram fraction measure lexical diversity and repetition, not semantic correctness.",
        "",
    ]
    best = max(rows, key=lambda r: r["chrf"])
    text += [
        f"Highest chrF among measured models: **{best['model']}**. BLEU, compute requirements and qualitative failures must also be considered."
    ]
    if missing:
        text += [
            "",
            "A four-model winner and encoder-decoder versus decoder-only comparison remain undetermined.",
        ]
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print("Completed:", status["completed_models"], "Missing:", missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--data", default="data")
    parser.add_argument("--report", default="RESULTS.md")
    args = parser.parse_args()
    main(args.out, args.data, args.report)
