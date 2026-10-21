"""Stage 1: symbolic supervision and evaluation references.

    python -m fuzzy_bisimulation.generate_labels --dataset fb15k-237

Produces ``data/<dataset>/labels/<tag>.pt`` with

* train: B*_s on the degree-aware training subgraph (paper, ``--train-reference
  induced``) or the exact local B^(k) around the training entities (``local``);
* val / test: references for held-out entities.  The default ``local`` mode
  gives full-graph-exact values without running Algorithm 1 on the whole graph
  (see local_sampling.py); ``full`` reproduces the paper's protocol and is only
  feasible for small graphs; ``induced`` is the cheap but biased option;
* wall-clock time of every offline stage, so evaluate.py can report the
  offline / online cost split instead of query time alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import DATA_ROOT
from .fuzzy_kg import CONFIDENCE_SCHEMES, DATASETS, dataset_folder, load_kg
from .exact_bisimulation import (SEMANTICS, TransitionSystem, build_state_labels,
                                 compute_bisimulation, label_similarity)
from .local_sampling import (induced_reference, local_reference, sample_pairs,
                             split_entities)


def add_label_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", required=True, help=f"one of {list(DATASETS)} or a folder in data/")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--confidence", default="auto", help=CONFIDENCE_SCHEMES)
    ap.add_argument("--label-mode", default="inout", choices=["none", "out", "in", "inout"],
                    help="fuzzy propositions used for L(s,t)")
    ap.add_argument("--semantics", default="godel", choices=list(SEMANTICS),
                    help="residuated pair (t-norm, implication); paper: godel")
    ap.add_argument("--discount", type=float, default=1.0,
                    help="lambda in F(R)=min(L,(1-l)+l*min(fwd,bwd)); paper: 1.0")
    ap.add_argument("--sample-ratio", type=float, default=0.1, help="training subgraph size |E_s|/|E|")
    ap.add_argument("--sampling", default="degree", choices=["degree", "random"])
    ap.add_argument("--gamma", type=float, default=0.7, help="degree-aware sampling bias")
    ap.add_argument("--train-reference", default="induced", choices=["induced", "local"])
    ap.add_argument("--eval-reference", default="local", choices=["local", "full", "induced"])
    ap.add_argument("--depth", type=int, default=4,
                    help="k for local references (exact full-graph B^(k))")
    ap.add_argument("--max-ball-nodes", type=int, default=20000,
                    help="cap on local ball size; depth shrinks if exceeded")
    ap.add_argument("--val-entities", type=int, default=500)
    ap.add_argument("--test-entities", type=int, default=300)
    ap.add_argument("--val-pairs", type=int, default=2000)
    ap.add_argument("--test-pairs", type=int, default=5000)
    ap.add_argument("--eps", type=float, default=1e-4, help="Algorithm 1 tolerance")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="optional safety cap; default: iterate until delta < eps (Algorithm 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None, help="output name (default: derived from options)")
    ap.add_argument("--force", action="store_true", help="recompute even if the file exists")


def label_tag(a: argparse.Namespace) -> str:
    sem = "" if (a.semantics, a.discount) == ("godel", 1.0) else f"_{a.semantics}{a.discount:g}"
    tag = (f"{a.confidence}_{a.label_mode}{sem}_{a.sampling}{a.sample_ratio:g}_g{a.gamma:g}"
           f"_tr-{a.train_reference}_ev-{a.eval_reference}{a.depth}_s{a.seed}")
    return tag.replace(":", "").replace(",", "-")


def label_path(a: argparse.Namespace) -> Path:
    return dataset_folder(a.dataset, a.data_root) / "labels" / f"{a.tag or label_tag(a)}.pt"


def stratified_pairs(B: torch.Tensor, n_pairs: int, seed: int,
                     edges: tuple[float, ...] = (0.0, 0.35, 0.7, 1.0001)) -> np.ndarray:
    """Pairs drawn evenly from value bins, so rare high-similarity pairs are
    represented (with Gödel semantics nearly all random pairs have B = 0)."""
    rng = np.random.default_rng(seed)
    iu = np.stack(np.triu_indices(B.shape[0], 1), 1)
    vals = B.numpy()[iu[:, 0], iu[:, 1]]
    per_bin = n_pairs // (len(edges) - 1)
    chosen = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((vals >= lo) & (vals < hi))
        chosen.append(rng.choice(idx, size=min(per_bin, len(idx)), replace=False))
    return iu[np.concatenate(chosen)]


def value_stats(v: np.ndarray, tau: float = 0.7) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
            "frac_zero": float((v == 0).mean()), "frac_pos": float((v >= tau).mean()),
            "quantiles": [float(x) for x in np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])]}


def _reference(mode: str, kg, labels, centers: np.ndarray, a, full_cache: dict) -> tuple[torch.Tensor, dict]:
    kw = dict(semantics=a.semantics, discount=a.discount)
    if mode == "local":
        ref = local_reference(kg.num_entities, kg.triples, kg.weights, kg.num_relations, labels, centers,
                              depth=a.depth, max_nodes=a.max_ball_nodes, eps=a.eps,
                              max_iter=a.max_iter, device=a.device, **kw)
        return ref.B, {"mode": "local", "exact_fixpoint": ref.exact_fixpoint, "depth": ref.depth,
                       "ball_nodes": ref.ball_nodes, "ball_triples": ref.ball_triples,
                       "iterations": ref.iterations, "seconds": ref.seconds}
    if mode == "induced":
        res, m = induced_reference(kg.triples, kg.weights, kg.num_relations, labels, centers,
                                   eps=a.eps, max_iter=a.max_iter, device=a.device, **kw)
        return res.B.cpu(), {"mode": "induced", "nodes": len(centers), "triples": m,
                             "iterations": res.iterations, "converged": res.converged,
                             "seconds": res.seconds}
    if mode == "full":
        if "B" not in full_cache:
            t0 = time.perf_counter()
            ts = TransitionSystem.from_triples(kg.num_entities, kg.triples, kg.weights,
                                               kg.num_relations, device=a.device)
            L = label_similarity(labels, kg.num_entities, a.device)
            res = compute_bisimulation(ts, L, eps=a.eps, max_iter=a.max_iter, verbose=True, **kw)
            full_cache["B"] = res.B
            full_cache["info"] = {"mode": "full", "nodes": kg.num_entities, "iterations": res.iterations,
                                  "converged": res.converged, "seconds": time.perf_counter() - t0}
            info = dict(full_cache["info"])
        else:
            info = dict(full_cache["info"], seconds=0.0, reused=True)
        c = torch.as_tensor(centers, device=full_cache["B"].device)
        return full_cache["B"][c][:, c].cpu(), info
    raise ValueError(mode)


def generate(a: argparse.Namespace) -> Path:
    out = label_path(a)
    if out.exists() and not a.force:
        print(f"labels exist: {out}")
        return out
    timing = {}
    t0 = time.perf_counter()
    kg = load_kg(a.dataset, a.data_root, confidence=a.confidence, seed=a.seed)
    timing["load_kg"] = time.perf_counter() - t0
    print(kg.summary())

    t0 = time.perf_counter()
    labels = build_state_labels(kg.num_entities, kg.triples, kg.weights, kg.num_relations, a.label_mode)
    split = split_entities(kg.degree(), a.sample_ratio, a.val_entities, a.test_entities,
                           a.sampling, a.gamma, a.seed)
    timing["sampling"] = time.perf_counter() - t0
    print(f"split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    full_cache: dict = {}
    print(f"[train] reference={a.train_reference}")
    B_train, train_info = _reference(a.train_reference, kg, labels, split.train, a, full_cache)
    timing["train_reference"] = train_info["seconds"]
    print(f"[val]   reference={a.eval_reference}")
    B_val, val_info = _reference(a.eval_reference, kg, labels, split.val, a, full_cache)
    timing["val_reference"] = val_info["seconds"]
    print(f"[test]  reference={a.eval_reference}")
    B_test, test_info = _reference(a.eval_reference, kg, labels, split.test, a, full_cache)
    timing["test_reference"] = test_info["seconds"]

    def pack(B: torch.Tensor, entities: np.ndarray, n_pairs: int, seed: int, info: dict) -> dict:
        uni = sample_pairs(len(entities), n_pairs, seed)
        strat = stratified_pairs(B, n_pairs, seed)
        return {"entities": entities, "B": B.to(torch.float32), "pairs_uniform": uni,
                "pairs_stratified": strat, "info": info,
                "stats_uniform": value_stats(B.numpy()[uni[:, 0], uni[:, 1]]),
                "stats_stratified": value_stats(B.numpy()[strat[:, 0], strat[:, 1]])}

    iu = torch.triu_indices(len(split.train), len(split.train), 1)
    train_vals = B_train[iu[0], iu[1]].numpy()
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "kg": {"name": kg.name, "entities": kg.num_entities, "relations": kg.num_relations,
               "triples": kg.num_triples, "confidence": a.confidence},
        "train": {"entities": split.train,
                  "B": B_train.to(torch.float16 if len(split.train) > 5000 else torch.float32),
                  "info": train_info, "stats": value_stats(train_vals)},
        "val": pack(B_val, split.val, a.val_pairs, a.seed + 11, val_info),
        "test": pack(B_test, split.test, a.test_pairs, a.seed + 12, test_info),
        "timing": timing,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    summary = {"train": payload["train"]["stats"], "train_info": train_info,
               "test_uniform": payload["test"]["stats_uniform"],
               "test_stratified": payload["test"]["stats_stratified"],
               "test_info": test_info, "timing": timing}
    print(json.dumps(summary, indent=2))
    for split_name, st in (("train", summary["train"]), ("test (uniform pairs)", summary["test_uniform"])):
        if st["frac_zero"] > 0.95 or st["std"] < 0.02:
            print(f"WARNING: {split_name} references are nearly constant "
                  f"(frac_zero={st['frac_zero']:.3f}, std={st['std']:.3f}). With Godel semantics a "
                  f"single unmatched transition gives 0; consider --discount < 1 or "
                  f"--semantics lukasiewicz, and report stratified-pair metrics.")
    print(f"saved {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_label_args(ap)
    generate(ap.parse_args())


if __name__ == "__main__":
    main()
