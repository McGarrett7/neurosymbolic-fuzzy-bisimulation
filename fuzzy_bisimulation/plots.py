"""Figures from the experiment suites' JSON results.

    python -m fuzzy_bisimulation.plots [--results results] [--out results/figures]

Reads every ``results/<suite>_<dataset>.json`` that exists and writes PNG + PDF
figures to ``--out``; suites that have not been run are skipped.

* ``training_loss_<ds>``      Fig. 1: train / val loss per epoch (main suite,
                              proposed model, first seed; read from its checkpoint)
* ``complexity_tradeoff_<ds>`` Fig. 2: offline runtime vs accuracy (1 - MAE), sampling suite
* ``main_<ds>``, ``ablation_<ds>``, ``encoders_<ds>``  bar charts, mean ± std over seeds
* ``alpha_<ds>``              metric vs fusion weight α
* ``sampling_<ds>``           MAE per sampling ratio and strategy
* ``latency``                 ns per query vs embedding dimension d
* ``stability_<ds>``          prediction change vs weight perturbation δ (Theorem 2)
* ``semantics_<ds>``          reference-value statistics per semantics / discount
* ``reference_<ds>``          local vs full-graph reference: error and time vs depth
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import CHECKPOINT_ROOT, RESULTS_ROOT  # noqa: E402

PAIRSET = "stratified"
COLORS = {"uniform": "#4C72B0", "stratified": "#DD8452"}


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"saved {out / name}.png / .pdf")


def _stats(values: list[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std())) if len(v) else (np.nan, np.nan)


def _find_checkpoint(path: str, roots: list[Path]) -> Path | None:
    """Checkpoint paths in the JSON may be absolute paths from another machine (Kaggle)."""
    p = Path(path)
    if p.exists():
        return p
    for root in roots:
        hits = list(root.rglob(p.name)) if root.exists() else []
        if hits:
            return hits[0]
    return None


def bar_grid(data: dict, title: str, out: Path, name: str, metrics=("mae", "rmse", "f1", "spearman")) -> None:
    """Grouped bars (uniform / stratified) per method for each metric."""
    rows = data["rows"]
    methods = list(rows)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.45 * len(methods) + 2.2), sharey=True)
    y = np.arange(len(methods))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        for k, ps in enumerate(("uniform", "stratified")):
            mu_sd = [_stats(rows[n][ps][m]) for n in methods]
            ax.barh(y + (k - 0.5) * 0.38, [a for a, _ in mu_sd], height=0.38, xerr=[b for _, b in mu_sd],
                    color=COLORS[ps], label=ps, capsize=2)
        ax.set_title(m.upper())
        ax.set_yticks(y, methods)
        ax.grid(axis="x", alpha=0.3)
    np.atleast_1d(axes)[0].invert_yaxis()  # shared y: invert once
    np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    _save(fig, out, name)


def plot_training_loss(data: dict, ds: str, out: Path, roots: list[Path]) -> None:
    """Fig. 1 from the proposed model's checkpoint history."""
    results = data.get("results", {})
    name = next((n for n in results if n.startswith("Proposed")), None)
    if not name or not results[name]:
        return
    ck = _find_checkpoint(results[name][0]["checkpoint"], roots)
    if ck is None:
        print(f"training_loss_{ds}: checkpoint {results[name][0]['checkpoint']} not found, skipped "
              "(put the checkpoints/ folder next to results/, or pass --checkpoints)")
        return
    hist = torch.load(ck, weights_only=False, map_location="cpu")["history"]
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["train_loss"] for h in hist], marker="o", ms=3, label="train MSE")
    ax.plot(ep, [h["val_loss"] for h in hist], marker="s", ms=3, label="validation MSE")
    best = int(np.argmin([h["val_loss"] for h in hist]))
    ax.axvline(ep[best], ls="--", c="grey", lw=1, label=f"best epoch ({ep[best]})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"Training convergence — {ds}")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out, f"training_loss_{ds}")


def plot_complexity_tradeoff(data: dict, ds: str, out: Path, reference: dict | None) -> None:
    """Fig. 2: offline cost vs 1 - MAE for the sampling variants."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strat, marker in (("random", "o"), ("degree", "s")):
        pts = []
        for n, r in data["rows"].items():
            m = re.match(r"(\d+)% (\w+)$", n)
            if not m or m.group(2) != strat:
                continue
            cost = data["extras"][n]["cost"]["offline_total"]
            pts.append((int(m.group(1)), cost, 1 - _stats(r[PAIRSET]["mae"])[0]))
        if pts:
            pts.sort()
            ax.plot([p[1] for p in pts], [p[2] for p in pts], marker=marker, label=f"{strat} sampling")
            for ratio, x, yv in pts:
                ax.annotate(f"{ratio}%", (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if reference:
        ax.axvline(reference["full"]["seconds"], ls="--", c="grey",
                   label=f"exact full-graph B* ({reference['full']['seconds']:.0f}s)")
    ax.set_xlabel("offline cost (s): sampling + reference + embeddings + training")
    ax.set_ylabel(f"accuracy 1 − MAE ({PAIRSET} pairs)")
    ax.set_title(f"Cost / accuracy trade-off — {ds}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, f"complexity_tradeoff_{ds}")


def plot_sampling(data: dict, ds: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for strat in ("random", "degree"):
        pts = sorted((int(m.group(1)), *_stats(r[PAIRSET]["mae"])) for n, r in data["rows"].items()
                     if (m := re.match(r"(\d+)% (\w+)$", n)) and m.group(2) == strat)
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o",
                        capsize=3, label=strat)
    ax.set_xlabel("sampling ratio (%)")
    ax.set_ylabel(f"MAE ({PAIRSET} pairs)")
    ax.set_title(f"Sampling sensitivity — {ds}")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out, f"sampling_{ds}")


def plot_alpha(data: dict, ds: str, out: Path) -> None:
    pts = sorted((float(n.split("=")[1]), r) for n, r in data["rows"].items()
                 if re.match(r"alpha=[\d.]+$", n))
    if not pts:
        return
    learned = data["rows"].get("alpha=learned")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, ("mae", "spearman")):
        for ps in ("uniform", "stratified"):
            mu_sd = [_stats(r[ps][m]) for _, r in pts]
            ax.errorbar([a for a, _ in pts], [a for a, _ in mu_sd], yerr=[b for _, b in mu_sd], marker="o",
                        capsize=3, color=COLORS[ps], label=ps)
            if learned:
                a_l = data["extras"]["alpha=learned"].get("alpha")
                if a_l is not None:
                    ax.scatter([a_l], [_stats(learned[ps][m])[0]], marker="*", s=150, color=COLORS[ps],
                               edgecolor="k", zorder=5)
        ax.axvline(0.7, ls=":", c="grey", lw=1)
        ax.set_xlabel("fusion weight α (★ = learned)")
        ax.set_ylabel(m.upper())
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Fusion-weight sensitivity — {ds}")
    _save(fig, out, f"alpha_{ds}")


def plot_latency(data: dict, out: Path) -> None:
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog([r["dim"] for r in rows], [r["seconds_per_query"] * 1e9 for r in rows], marker="o")
    ax.set_xlabel("embedding dimension d")
    ax.set_ylabel("ns per query")
    ax.set_title("Query cost on precomputed embeddings: O(d)")
    ax.grid(alpha=0.3, which="both")
    _save(fig, out, "latency")


def plot_stability(data: dict, ds: str, out: Path) -> None:
    rows = data["rows"]
    d = [r["delta"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d, [r["mean"] for r in rows], marker="o", label="mean |ΔB̂|")
    ax.plot(d, [r["max"] for r in rows], marker="s", label="max |ΔB̂|")
    ax.plot(d, d, ls="--", c="grey", label="y = δ")
    ax.set_xlabel("weight perturbation δ")
    ax.set_ylabel("prediction change")
    ax.set_title(f"Stability under perturbation (Theorem 2) — {ds}")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out, f"stability_{ds}")


def plot_semantics(data: dict, ds: str, out: Path) -> None:
    names = list(data)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 4))
    for k, (key, label) in enumerate((("mean", "mean B*"), ("std", "std"), ("frac_zero", "fraction = 0"),
                                      ("frac_pos", "fraction ≥ 0.7"))):
        ax.bar(x + (k - 1.5) * 0.2, [data[n]["test_uniform"][key] for n in names], width=0.2, label=label)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Reference values per semantics / discount (test, uniform pairs) — {ds}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, f"semantics_{ds}")


def plot_reference(data: dict, ds: str, out: Path) -> None:
    loc = data["local"]
    dep = [r["depth"] for r in loc]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(dep, [max(r["mean_abs"], 1e-12) for r in loc], marker="o", label="mean |B^(k) − B*|")
    axes[0].semilogy(dep, [max(r["max_abs"], 1e-12) for r in loc], marker="s", label="max |B^(k) − B*|")
    axes[0].set_xlabel("local depth k")
    axes[0].set_ylabel("error vs full-graph B*")
    axes[0].legend()
    axes[1].plot(dep, [r["seconds"] for r in loc], marker="o", label="local reference")
    axes[1].axhline(data["full"]["seconds"], ls="--", c="grey", label="full-graph Algorithm 1")
    axes[1].set_xlabel("local depth k")
    axes[1].set_ylabel("seconds")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Local vs full-graph reference — {ds}")
    _save(fig, out, f"reference_{ds}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/figures")
    ap.add_argument("--checkpoints", type=Path, default=None, help="default: <results>/../checkpoints")
    a = ap.parse_args()
    out = a.out or a.results / "figures"
    roots = [r for r in (a.checkpoints, a.results.parent / "checkpoints", CHECKPOINT_ROOT) if r is not None]
    load = lambda f: json.loads(f.read_text())  # noqa: E731
    files = sorted(a.results.glob("*.json"))
    if not files:
        print(f"no results in {a.results}")
        return
    for f in files:
        suite, _, ds = f.stem.partition("_")
        if ds.endswith("zeroshot"):
            continue
        try:
            data = load(f)
            if suite == "main":
                bar_grid(data, f"Main comparison — {ds}", out, f"main_{ds}")
                plot_training_loss(data, ds, out, roots)
            elif suite == "ablation":
                bar_grid(data, f"Ablation — {ds}", out, f"ablation_{ds}")
            elif suite == "encoders":
                bar_grid(data, f"Semantic encoders — {ds}", out, f"encoders_{ds}")
            elif suite in ("confidence", "train-ref"):
                bar_grid(data, f"{suite} — {ds}", out, f"{suite}_{ds}")
            elif suite == "alpha":
                plot_alpha(data, ds, out)
            elif suite == "sampling":
                plot_sampling(data, ds, out)
                ref = a.results / f"reference_{ds}.json"
                plot_complexity_tradeoff(data, ds, out, load(ref) if ref.exists() else None)
            elif suite == "latency":
                plot_latency(data, out)
            elif suite == "stability":
                plot_stability(data, ds, out)
            elif suite == "semantics":
                plot_semantics(data, ds, out)
            elif suite == "reference":
                plot_reference(data, ds, out)
        except Exception as exc:  # one malformed file should not stop the others
            print(f"{f.name}: skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
