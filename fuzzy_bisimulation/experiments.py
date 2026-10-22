"""Experiment suites of the paper (Section VI) and of the reviewer requests.

    python -m fuzzy_bisimulation.experiments <suite> --dataset <name> [options]

Paper:
  main          Table V: proposed vs trained / zero-shot baselines, 5 seeds,
                paired t-tests with Benjamini-Hochberg correction (Table VI)
  ablation      Table VIII
  sampling      Table VII: sampling ratio x strategy, plus a gamma sweep
  latency       Fig. 2 companion: query cost vs embedding dimension

Reviewer requests:
  reference     local exact references vs full-graph B* (accuracy and time),
                and the sampled-subgraph gap eps_s of Assumption 1
  train-ref     paper's induced-subgraph supervision vs local exact supervision
  confidence    sensitivity to the confidence assumption (Beta(2,2) vs other
                synthetic and data-derived confidences)
  alpha         fusion-weight sensitivity, including a learned alpha
  encoders      semantic encoder comparison (OpenAI vs open models vs TF-IDF)
  semantics     reference-value distribution under Gödel / Łukasiewicz /
                product semantics and discounts (no training)
  stability     Theorem 2: prediction change under weight perturbation delta
  assumptions   empirical eps_s, eps_nn, Theorem 1 bound and Lipschitz constant

Every suite writes results/<suite>_<dataset>.json and a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import RESULTS_ROOT
from .fuzzy_kg import load_kg, perturb_weights
from .evaluate import (assumption_checks, baseline_scores, evaluate_checkpoint,
                       latency_vs_dimension, paired_tests)
from .exact_bisimulation import (TransitionSystem, build_state_labels, compute_bisimulation,
                                 label_similarity)
from .generate_labels import add_label_args, generate, value_stats
from .local_sampling import induced_reference, local_reference, split_entities
from .train import DEFAULT_SEMANTIC, add_train_args, eval_pairs, load_model, train

METRICS = ("mae", "rmse", "f1", "spearman")


# --------------------------------------------------------------------------- helpers

def label_args(common: argparse.Namespace, **over) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_label_args(ap)
    a = ap.parse_args(["--dataset", common.dataset])
    for k in ("confidence", "semantics", "discount", "train_reference", "depth", "device",
              "sample_ratio", "label_mode"):
        if getattr(common, k, None) is not None:
            setattr(a, k, getattr(common, k))
    for k, v in over.items():
        setattr(a, k, v)
    return a


def train_args(common: argparse.Namespace, labels: Path, **over) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_train_args(ap)
    a = ap.parse_args(["--labels", str(labels)])
    a.semantic, a.epochs, a.steps_per_epoch = common.semantic, common.epochs, common.steps_per_epoch
    a.train_pairs = common.train_pairs
    a.pair_sampling, a.device, a.quiet = common.pair_sampling, common.device, True
    for k, v in over.items():
        setattr(a, k, v)
    return a


def run_method(common, seed: int, label_over: dict, train_over: dict) -> dict:
    labels = generate(label_args(common, seed=seed, **label_over))
    ckpt = train(train_args(common, labels, seed=seed, **train_over))
    return evaluate_checkpoint(ckpt, common.device)


def collect(results: list[dict], split: str = "test") -> dict:
    """{pairset: {metric: [values over seeds]}}"""
    out = {}
    for which in ("uniform", "stratified"):
        out[which] = {m: [r[split][which].get(m, float("nan")) for r in results] for m in METRICS}
    return out


def mean_std(values: list[float]) -> str:
    v = np.asarray(values, dtype=float)
    return f"{np.nanmean(v):.3f}±{np.nanstd(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def table(rows: dict[str, dict], pairset: str, extra: dict[str, dict] | None = None) -> str:
    head = f"| Method | {' | '.join(m.upper() for m in METRICS)} |"
    lines = [f"**{pairset} pairs**", "", head, "|" + "---|" * (len(METRICS) + 1)]
    for name, col in rows.items():
        lines.append(f"| {name} | " + " | ".join(mean_std(col[pairset][m]) for m in METRICS) + " |")
    return "\n".join(lines) + "\n"


def save(common, suite: str, payload: dict, markdown: str) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_ROOT / f"{suite}_{common.dataset}"
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=float))
    stem.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"saved {stem}.json / .md")


def run_grid(common, variants: dict[str, tuple[dict, dict]], suite: str,
             reference: str | None = None) -> None:
    raw, rows = {}, {}
    for name, (lo, to) in variants.items():
        print(f"=== {suite}: {name}")
        try:
            raw[name] = [run_method(common, s, lo, to) for s in common.seeds]
        except Exception as exc:                      # e.g. optional encoder not installed
            traceback.print_exc()
            print(f"skipping {name}: {exc}")
            continue
        rows[name] = collect(raw[name])
    md = f"# {suite} — {common.dataset}\n\n" + table(rows, "uniform") + "\n" + table(rows, "stratified")
    tests = {}
    if reference in rows and len(common.seeds) > 1:
        for name in rows:
            if name != reference:
                tests[name] = {w: paired_tests(rows[reference][w], rows[name][w]) for w in ("uniform", "stratified")}
        md += f"\nPaired t-tests vs {reference} (BH-corrected p-values, stratified pairs):\n\n"
        for name, t in tests.items():
            md += f"- {name}: " + ", ".join(f"{m} p={v['p_bh']:.2e}" for m, v in t["stratified"].items()) + "\n"
    extras = {n: {k: r[0].get(k) for k in ("cost", "alpha")} for n, r in raw.items()}
    save(common, suite, {"rows": rows, "tests": tests, "extras": extras,
                         "results": {n: [{k: v for k, v in r.items() if k != "config"} for r in rs]
                                     for n, rs in raw.items()}}, md)


# --------------------------------------------------------------------------- suites

def suite_main(common) -> None:
    variants = {
        "Proposed (R-GCN + semantic, a=0.7)": ({}, {}),
        "R-GCN": ({}, {"alpha": "1.0"}),
        "GCN": ({}, {"encoder": "gcn", "alpha": "1.0"}),
        "GraphSAGE": ({}, {"encoder": "sage", "alpha": "1.0"}),
        "MLP on structural features": ({}, {"encoder": "mlp", "alpha": "1.0"}),
        "Semantic only (trained projection)": ({}, {"alpha": "0.0"}),
    }
    run_grid(common, variants, "main", reference="Proposed (R-GCN + semantic, a=0.7)")
    labels = generate(label_args(common, seed=common.seeds[0]))
    zs = baseline_scores(labels, common.semantic)
    print("zero-shot baselines (seed 0):")
    for name, r in zs.items():
        print(f"  {name}: {r['test']['stratified']}")
    (RESULTS_ROOT / f"main_{common.dataset}_zeroshot.json").write_text(json.dumps(zs, indent=2, default=float))


def suite_ablation(common) -> None:
    variants = {
        "Full model": ({}, {}),
        "w/o semantic (a=1)": ({}, {"alpha": "1.0"}),
        "w/o graph learning (a=0)": ({}, {"alpha": "0.0"}),
        "w/o symbolic supervision (Jaccard target)": ({}, {"target": "jaccard"}),
        "random sampling": ({"sampling": "random"}, {}),
        "with fuzzy edge weights in messages": ({}, {"edge_weight": True}),
        "paper's final ReLU (full model)": ({}, {"final_relu": True}),
        "paper's final ReLU, w/o semantic (a=1)": ({}, {"final_relu": True, "alpha": "1.0"}),
    }
    run_grid(common, variants, "ablation", reference="Full model")


def suite_sampling(common) -> None:
    variants = {}
    for ratio in (0.05, 0.1, 0.2):
        for strat in ("random", "degree"):
            variants[f"{ratio:.0%} {strat}"] = ({"sample_ratio": ratio, "sampling": strat}, {})
    for gamma in (0.0, 0.3, 0.5, 1.0):
        variants[f"10% degree gamma={gamma}"] = ({"gamma": gamma}, {})
    run_grid(common, variants, "sampling")


def suite_alpha(common) -> None:
    variants = {f"alpha={a:.1f}": ({}, {"alpha": f"{a:.1f}"}) for a in np.linspace(0, 1, 11)}
    variants["alpha=learned"] = ({}, {"alpha": "learn"})
    run_grid(common, variants, "alpha")


def suite_encoders(common) -> None:
    variants = {f"{e}": ({}, {"semantic": e}) for e in common.encoders}
    run_grid(common, variants, "encoders")


def suite_train_ref(common) -> None:
    variants = {"induced subgraph B*_s (paper)": ({"train_reference": "induced"}, {}),
                "local exact B^(k)": ({"train_reference": "local"}, {})}
    run_grid(common, variants, "train-ref")


def suite_confidence(common) -> None:
    kg = load_kg(common.dataset)
    schemes = ["beta:2,2", "beta:0.5,0.5", "beta:5,1", "uniform", "crisp", "pathsupport", "relfreq"]
    if kg.given_weights is not None:
        schemes.insert(0, "given")
    variants = {s: ({"confidence": s}, {}) for s in schemes}
    run_grid(common, variants, "confidence")


def suite_semantics(common) -> None:
    combos = [("godel", 1.0), ("godel", 0.9), ("godel", 0.7), ("godel", 0.5),
              ("lukasiewicz", 1.0), ("lukasiewicz", 0.9), ("lukasiewicz", 0.7), ("product", 1.0)]
    rows, md = {}, [f"# semantics — {common.dataset}\n",
                    "| semantics | discount | mean | std | frac zero | frac >= 0.7 | iterations | seconds |",
                    "|---|---|---|---|---|---|---|---|"]
    for sem, disc in combos:
        a = label_args(common, semantics=sem, discount=disc, seed=common.seeds[0])
        payload = torch.load(generate(a), weights_only=False)
        st, info = payload["test"]["stats_uniform"], payload["test"]["info"]
        rows[f"{sem}/{disc}"] = {"test_uniform": st, "train": payload["train"]["stats"], "info": info}
        md.append(f"| {sem} | {disc} | {st['mean']:.3f} | {st['std']:.3f} | {st['frac_zero']:.3f} | "
                  f"{st['frac_pos']:.3f} | {info['iterations']} | {info['seconds']:.2f} |")
    save(common, "semantics", rows, "\n".join(md) + "\n")


def suite_reference(common) -> None:
    """Accuracy and cost of local references against the full-graph fixpoint,
    and the gap between induced-subgraph labels and full-graph values."""
    a = label_args(common, seed=common.seeds[0])
    kg = load_kg(a.dataset, a.data_root, confidence=a.confidence, seed=a.seed)
    n = kg.num_entities
    labels = build_state_labels(n, kg.triples, kg.weights, kg.num_relations, a.label_mode)
    kw = dict(semantics=a.semantics, discount=a.discount)
    t0 = time.perf_counter()
    ts = TransitionSystem.from_triples(n, kg.triples, kg.weights, kg.num_relations, device=a.device)
    full = compute_bisimulation(ts, label_similarity(labels, n, a.device), eps=a.eps,
                                max_iter=a.max_iter, verbose=True, **kw)
    full_seconds = time.perf_counter() - t0
    split = split_entities(kg.degree(), a.sample_ratio, a.val_entities, a.test_entities, a.sampling,
                           a.gamma, a.seed)
    c = torch.as_tensor(split.test, device=full.B.device)
    ref_full = full.B[c][:, c].cpu()
    iu = torch.triu_indices(len(c), len(c), 1)
    out = {"full": {"seconds": full_seconds, "iterations": full.iterations, "nodes": n}, "local": [],
           "induced_gap": []}
    md = [f"# reference — {common.dataset}\n",
          f"Full-graph Algorithm 1: {full_seconds:.2f}s, {full.iterations} iterations, n={n}\n",
          "| local depth | ball nodes | exact B* | seconds | max abs diff to B* | mean abs diff |",
          "|---|---|---|---|---|---|"]
    for depth in range(1, common.max_depth + 1):
        ref = local_reference(n, kg.triples, kg.weights, kg.num_relations, labels, split.test, depth=depth,
                              max_nodes=None, eps=a.eps, max_iter=a.max_iter, device=a.device, **kw)
        d = (ref.B - ref_full)[iu[0], iu[1]].abs()
        row = {"depth": depth, "ball_nodes": ref.ball_nodes, "exact_fixpoint": ref.exact_fixpoint,
               "seconds": ref.seconds, "max_abs": float(d.max()), "mean_abs": float(d.mean())}
        out["local"].append(row)
        md.append(f"| {depth} | {ref.ball_nodes} | {ref.exact_fixpoint} | {ref.seconds:.2f} | "
                  f"{row['max_abs']:.4f} | {row['mean_abs']:.4f} |")
    md += ["", "Assumption 1 (eps_s): induced-subgraph labels B*_s vs full-graph B* on the same entities", "",
           "| sampling | ratio | n_s | E abs(B*_s - B*) | mean B*_s | mean B* |", "|---|---|---|---|---|---|"]
    for strat in ("random", "degree"):
        for ratio in (0.05, 0.1, 0.2):
            sp_ = split_entities(kg.degree(), ratio, 0, 0, strat, a.gamma, a.seed)
            res, _ = induced_reference(kg.triples, kg.weights, kg.num_relations, labels, sp_.train,
                                       eps=a.eps, max_iter=a.max_iter, device=a.device, **kw)
            t = torch.as_tensor(sp_.train, device=full.B.device)
            jj = torch.triu_indices(len(t), len(t), 1)
            bs = res.B.cpu()[jj[0], jj[1]]
            bf = full.B[t][:, t].cpu()[jj[0], jj[1]]
            row = {"sampling": strat, "ratio": ratio, "n_s": len(t), "eps_s": float((bs - bf).abs().mean()),
                   "mean_sub": float(bs.mean()), "mean_full": float(bf.mean())}
            out["induced_gap"].append(row)
            md.append(f"| {strat} | {ratio:.0%} | {len(t)} | {row['eps_s']:.4f} | {row['mean_sub']:.3f} | "
                      f"{row['mean_full']:.3f} |")
    save(common, "reference", out, "\n".join(md) + "\n")


def suite_latency(common) -> None:
    rows = latency_vs_dimension()
    md = ["# latency — cosine query on precomputed embeddings\n", "| d | ns / query |", "|---|---|"]
    md += [f"| {r['dim']} | {r['seconds_per_query'] * 1e9:.1f} |" for r in rows]
    save(common, "latency", {"rows": rows}, "\n".join(md) + "\n")


def suite_stability(common) -> None:
    labels = generate(label_args(common, seed=common.seeds[0]))
    ckpt = train(train_args(common, labels, seed=common.seeds[0]))
    payload = torch.load(labels, weights_only=False)
    pairs, _ = eval_pairs(payload["test"], "both")
    p = torch.as_tensor(pairs)
    model, g, _, _ = load_model(ckpt)
    with torch.no_grad():
        base = model.similarity(*(model.embed(g)[p[:, k]] for k in (0, 1)))
    kg = load_kg(payload["config"]["dataset"], Path(payload["config"]["data_root"]),
                 confidence=payload["config"]["confidence"], seed=payload["config"]["seed"])
    rows, md = [], [f"# stability — {common.dataset}\n",
                    "| delta | mean abs change | max abs change | max / delta |", "|---|---|---|---|"]
    for delta in (0.01, 0.02, 0.05, 0.1, 0.2):
        m2, g2, _, _ = load_model(ckpt, weights=perturb_weights(kg.weights, delta, seed=1))
        with torch.no_grad():
            pert = m2.similarity(*(m2.embed(g2)[p[:, k]] for k in (0, 1)))
        d = (pert - base).abs()
        rows.append({"delta": delta, "mean": float(d.mean()), "max": float(d.max())})
        md.append(f"| {delta} | {d.mean():.4f} | {d.max():.4f} | {d.max() / delta:.2f} |")
    save(common, "stability", {"rows": rows}, "\n".join(md) + "\n")


def suite_assumptions(common) -> None:
    rows, md = {}, [f"# assumptions — {common.dataset}\n",
                    "| supervision | seed | eps_s | eps_nn | bound eps_s+eps_nn | train error | test error | "
                    "L_phi spectral | L_phi max observed |", "|---|---|---|---|---|---|---|---|---|"]
    for ref in ("induced", "local"):
        for seed in common.seeds:
            labels = generate(label_args(common, seed=seed, train_reference=ref))
            ckpt = train(train_args(common, labels, seed=seed))
            r = assumption_checks(ckpt, common.device)
            rows[f"{ref}/s{seed}"] = r
            lip = r.get("lipschitz_projection", {})
            md.append(f"| {ref} | {seed} | {r['eps_s']:.4f} | {r['eps_nn']:.4f} | "
                      f"{r['theorem1_bound_eps_s_plus_eps_nn']:.4f} | {r['train_total_error']:.4f} | "
                      f"{r['test_total_error']:.4f} | {lip.get('spectral_norm', float('nan')):.3f} | "
                      f"{lip.get('empirical_max', float('nan')):.3f} |")
    md.append("\nThe bound holds on training pairs by the triangle inequality; the test column shows "
              "how far it is from held-out error (Theorem 1 says nothing about unseen entities). "
              "Theorem 2's constant C is measured by the `stability` suite.")
    save(common, "assumptions", rows, "\n".join(md) + "\n")


SUITES = {"main": suite_main, "ablation": suite_ablation, "sampling": suite_sampling,
          "alpha": suite_alpha, "encoders": suite_encoders, "train-ref": suite_train_ref,
          "confidence": suite_confidence, "semantics": suite_semantics, "reference": suite_reference,
          "latency": suite_latency, "stability": suite_stability, "assumptions": suite_assumptions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=list(SUITES))
    ap.add_argument("--dataset", default="fb15k-237", choices=["fb15k-237", "wn18rr", "yago3-10"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--semantic", default=DEFAULT_SEMANTIC)
    ap.add_argument("--encoders", nargs="+",
                    default=["tfidf", "st:all-MiniLM-L6-v2", "st:all-mpnet-base-v2",
                             "openai:text-embedding-3-large:768"])
    ap.add_argument("--confidence", default=None)
    ap.add_argument("--semantics", default=None)
    ap.add_argument("--discount", type=float, default=None)
    ap.add_argument("--train-reference", default=None, choices=["induced", "local"])
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=6, help="reference suite: deepest local ball")
    ap.add_argument("--sample-ratio", type=float, default=None)
    ap.add_argument("--label-mode", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0: full pass over P per epoch (paper)")
    ap.add_argument("--train-pairs", type=int, default=0, help="|P|; 0: all training-entity pairs")
    ap.add_argument("--pair-sampling", default="uniform", choices=["uniform", "balanced"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common = ap.parse_args()
    SUITES[common.suite](common)


if __name__ == "__main__":
    main()
