"""Evaluation: accuracy, cost split, query-latency scaling, baselines, statistics.

    python -m fuzzy_bisimulation.evaluate --checkpoint checkpoints/<run>.pt
    python -m fuzzy_bisimulation.evaluate --labels data/<ds>/labels/<tag>.pt --baselines
    python -m fuzzy_bisimulation.evaluate --latency

Reported per run:
* MAE, RMSE, F1 at tau = 0.7 (Section V-C), plus precision / recall, Pearson,
  Spearman and the positive rate, on uniform pairs and on value-stratified
  pairs (uniform pairs alone can be almost all zeros under Gödel semantics).
* Cost split (reviewer comments on scalability and "constant time"): offline
  stages (sampling, reference labels, semantic embeddings, training,
  embedding construction) versus online per-query time, the measured time of
  the exact reference for the same entities, and the break-even number of
  queries after which the learned model is cheaper end to end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from . import RESULTS_ROOT
from .exact_bisimulation import build_state_labels
from .local_sampling import local_reference, sample_pairs
from .semantic import load_or_encode
from .model import structural_features
from .train import DEFAULT_SEMANTIC, eval_pairs, kg_for_labels, load_model


# --------------------------------------------------------------------------- metrics

def regression_metrics(y: np.ndarray, p: np.ndarray, tau: float = 0.7) -> dict:
    """Accuracy metrics of predicted against reference degrees.

    F1 binarises both at ``tau`` (Section V-C).

    Args:
        y: Reference values.
        p: Predictions.
        tau: Threshold for precision / recall / F1.

    Returns:
        Dict with ``mae``, ``rmse``, ``f1``, ``precision``, ``recall``,
        ``pos_rate``, ``confusion`` and, when both inputs vary, ``pearson`` and
        ``spearman``.
    """
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    yt, pt = y >= tau, p >= tau
    tp, fp, fn = int((yt & pt).sum()), int((~yt & pt).sum()), int((yt & ~pt).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"n": int(len(y)), "mae": float(np.abs(y - p).mean()),
           "rmse": float(np.sqrt(((y - p) ** 2).mean())),
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "precision": prec, "recall": rec, "pos_rate": float(yt.mean()),
           "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": int((~yt & ~pt).sum())}}
    if y.std() > 0 and p.std() > 0:
        out["pearson"] = float(stats.pearsonr(y, p)[0])
        out["spearman"] = float(stats.spearmanr(y, p)[0])
    return out


def score_both(part: dict, predict) -> dict:
    """Metrics on the uniform and stratified pair sets of a val/test block.

    Args:
        part: The ``"val"`` or ``"test"`` block of a label file.
        predict: Function mapping global entity pairs (p, 2) to predictions (p,).

    Returns:
        ``{"uniform": metrics, "stratified": metrics}``.
    """
    res = {}
    for which in ("uniform", "stratified"):
        pairs, y = eval_pairs(part, which)
        res[which] = regression_metrics(y, predict(pairs))
    return res


# --------------------------------------------------------------------------- learned model

def evaluate_checkpoint(ckpt_path: Path, device: str = "cpu") -> dict:
    """Evaluate a trained model: accuracy on val/test and the cost report.

    Args:
        ckpt_path: Checkpoint file.
        device: Torch device.

    Returns:
        Dict with configs, α, ``test`` / ``val`` metrics, reference statistics
        and ``cost``.
    """
    model, g, ckpt, payload = load_model(ckpt_path, device)
    t0 = time.perf_counter()
    with torch.no_grad():
        z = model.embed(g)
    embed_seconds = time.perf_counter() - t0

    def predict(pairs: np.ndarray) -> np.ndarray:
        """Predict degrees for global entity pairs from the precomputed embeddings.

        Args:
            pairs: Int array (p, 2).

        Returns:
            Predictions (p,).
        """
        p = torch.as_tensor(pairs, device=device)
        with torch.no_grad():
            return model.similarity(z[p[:, 0]], z[p[:, 1]]).cpu().numpy()

    result = {"checkpoint": str(ckpt_path), "config": ckpt["config"],
              "label_config": payload["config"], "alpha": float(model.alpha),
              "test": score_both(payload["test"], predict),
              "val": score_both(payload["val"], predict),
              "train_reference_stats": payload["train"]["stats"],
              "test_reference_info": payload["test"]["info"]}
    result["cost"] = cost_report(payload, ckpt, z, embed_seconds)
    return result


def measure_query_seconds(z: torch.Tensor, n_queries: int = 20000, seed: int = 0) -> dict:
    """Time per pairwise query on precomputed embeddings.

    Args:
        z: Embeddings (n, d).
        n_queries: Number of random pairs.
        seed: Random seed.

    Returns:
        ``{"batched": s, "single": s}``: seconds per query when vectorised and
        when issued one by one.
    """
    gen = torch.Generator().manual_seed(seed)
    i = torch.randint(0, len(z), (n_queries,), generator=gen).to(z.device)
    j = torch.randint(0, len(z), (n_queries,), generator=gen).to(z.device)
    with torch.no_grad():
        t0 = time.perf_counter()
        F.cosine_similarity(z[i], z[j], dim=-1)
        batched = (time.perf_counter() - t0) / n_queries
        k = min(2000, n_queries)
        t0 = time.perf_counter()
        for a in range(k):
            F.cosine_similarity(z[i[a]], z[j[a]], dim=0)
        single = (time.perf_counter() - t0) / k
    return {"batched": batched, "single": single}


def cost_report(payload: dict, ckpt: dict, z: torch.Tensor, embed_seconds: float) -> dict:
    """Offline / online cost split and the break-even number of queries.

    Args:
        payload: Loaded label file (stage-1 timings).
        ckpt: Loaded checkpoint (training and semantic timings).
        z: Precomputed embeddings.
        embed_seconds: Time to compute ``z``.

    Returns:
        Dict with offline stage times, per-query times, the exact-reference cost
        for the test entities and ``break_even_queries`` (None if the learned
        query is not cheaper per pair).
    """
    t = payload["timing"]
    offline = {"sampling": t["sampling"], "train_reference": t["train_reference"],
               "semantic_embedding": ckpt["timing"]["semantic_embedding"],
               "training": ckpt["timing"]["train"], "embedding_construction": embed_seconds}
    offline_total = float(sum(offline.values()))
    q = measure_query_seconds(z)
    n_test = len(payload["test"]["entities"])
    test_pairs = n_test * (n_test - 1) // 2
    exact_seconds = payload["test"]["info"]["seconds"]
    exact_per_pair = exact_seconds / max(test_pairs, 1)
    learned_per_pair = q["batched"]
    gain = exact_per_pair - learned_per_pair
    return {
        "offline_seconds": offline, "offline_total": offline_total,
        "query_seconds": q, "embedding_dim": int(z.shape[1]),
        "exact_reference": {"mode": payload["test"]["info"]["mode"], "seconds": exact_seconds,
                            "entities": n_test, "pairs": test_pairs, "per_pair": exact_per_pair,
                            "ball_nodes": payload["test"]["info"].get("ball_nodes")},
        # Queries needed before offline cost + learned queries < exact queries.
        "break_even_queries": float(offline_total / gain) if gain > 0 else None,
        "note": "per-query cost is O(d) for fixed precomputed embeddings; it is independent of "
                "|E| only under that assumption, and excludes the offline stages listed here",
    }


# --------------------------------------------------------------------------- zero-shot baselines

def baseline_scores(labels_path: Path, semantic: str = "tfidf") -> dict:
    """Scores of methods that need no training on bisimulation targets.

    Structural-feature cosine, semantic cosine ("LLM similarity") and
    neighbourhood Jaccard.

    Args:
        labels_path: Label file.
        semantic: Encoder spec for the semantic cosine.

    Returns:
        ``{method: {"test": score_both(...)}}``.
    """
    payload = torch.load(labels_path, weights_only=False)
    kg = kg_for_labels(payload)
    out = {}
    sem, _ = load_or_encode(kg, semantic, Path(payload["config"]["data_root"]))
    embeddings = {"feature_cosine": torch.as_tensor(structural_features(kg))}
    if sem is not None:
        embeddings[f"semantic_cosine[{semantic}]"] = torch.as_tensor(sem)   # "LLM Similarity"
    for name, E in embeddings.items():
        def predict(pairs, E=E):
            """Cosine similarity of embedding rows mapped to [0, 1].

            Args:
                pairs: Global entity pairs (p, 2).
                E: Embedding matrix (bound per method).

            Returns:
                Predictions (p,).
            """
            p = torch.as_tensor(pairs)
            return ((F.cosine_similarity(E[p[:, 0]], E[p[:, 1]], dim=-1) + 1) / 2).numpy()
        out[name] = {"test": score_both(payload["test"], predict)}

    import scipy.sparse as sp
    n, R = kg.num_entities, kg.num_relations
    h, r, t = kg.triples.T
    X = sp.csr_matrix((np.ones(2 * len(h), dtype=np.float32),
                       (np.concatenate([h, t]), np.concatenate([r * n + t, (R + r) * n + h]))),
                      shape=(n, 2 * R * n))
    X.data[:] = 1.0
    size = np.asarray(X.sum(1)).ravel()

    def jaccard(pairs):
        """Neighbourhood Jaccard over (direction, relation, neighbour) keys.

        Args:
            pairs: Global entity pairs (p, 2).

        Returns:
            Similarities (p,).
        """
        inter = np.asarray(X[pairs[:, 0]].multiply(X[pairs[:, 1]]).sum(1)).ravel()
        union = size[pairs[:, 0]] + size[pairs[:, 1]] - inter
        return np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    out["neighbour_jaccard"] = {"test": score_both(payload["test"], jaccard)}
    return out


# --------------------------------------------------------------------------- assumptions

def assumption_checks(ckpt_path: Path, device: str = "cpu", n_pairs: int = 20000) -> dict:
    """Empirical counterparts of Assumptions 1-3 and Theorem 1 (Section IV-G).

    * eps_s: E|B*_s - B| on training pairs, B = exact local full-graph
      reference (0 by construction when the labels already are local references).
    * eps_nn: E|B̂ - B*_s| on training pairs (fit to the supervision).
    * total: E|B̂ - B| on training pairs and on held-out test pairs.
    * L_phi: Lipschitz constant of the semantic projection, as the spectral
      norm of W_proj (a global bound) and the largest observed ratio
      ||ϕ_proj(a) - ϕ_proj(b)|| / ||ϕ(a) - ϕ(b)||.  Assumption 3 as printed
      compares against ||e_i - e_j||, which is undefined for entities; ϕ is the
      meaningful input space.

    Args:
        ckpt_path: Checkpoint file.
        device: Torch device.
        n_pairs: Number of sampled pairs.

    Returns:
        Dict with ``eps_s``, ``eps_nn``, train / test errors, the Theorem 1 bound,
        the generalisation gap and, with a semantic channel,
        ``lipschitz_projection``.
    """
    model, g, ckpt, payload = load_model(ckpt_path, device)
    cfg = payload["config"]
    with torch.no_grad():
        z = model.embed(g)
    ent = payload["train"]["entities"]
    B_s = payload["train"]["B"].float().numpy()
    pairs = sample_pairs(len(ent), n_pairs, seed=0)
    pi = torch.as_tensor(ent[pairs[:, 0]], device=device)
    pj = torch.as_tensor(ent[pairs[:, 1]], device=device)
    with torch.no_grad():
        pred = model.similarity(z[pi], z[pj]).cpu().numpy()
    y_s = B_s[pairs[:, 0], pairs[:, 1]]
    out = {"eps_nn": float(np.abs(pred - y_s).mean()), "train_reference": cfg["train_reference"]}

    if cfg["train_reference"] == "local":
        y = y_s
        out["eps_s"] = 0.0
    else:
        kg = kg_for_labels(payload)
        labels = build_state_labels(kg.num_entities, kg.triples, kg.weights, kg.num_relations,
                                    cfg["label_mode"])
        ref = local_reference(kg.num_entities, kg.triples, kg.weights, kg.num_relations, labels, ent,
                              depth=cfg["depth"], max_nodes=cfg["max_ball_nodes"], eps=cfg["eps"],
                              max_iter=cfg["max_iter"], device=device, semantics=cfg["semantics"],
                              discount=cfg["discount"])
        y = ref.B.numpy()[pairs[:, 0], pairs[:, 1]]
        out["eps_s"] = float(np.abs(y_s - y).mean())
        out["eps_s_reference"] = {"depth": ref.depth, "exact_fixpoint": ref.exact_fixpoint,
                                  "ball_nodes": ref.ball_nodes, "seconds": ref.seconds}
    out["train_total_error"] = float(np.abs(pred - y).mean())
    out["theorem1_bound_eps_s_plus_eps_nn"] = out["eps_s"] + out["eps_nn"]
    tp, ty = eval_pairs(payload["test"], "uniform")
    tpt = torch.as_tensor(tp, device=device)
    with torch.no_grad():
        tpred = model.similarity(z[tpt[:, 0]], z[tpt[:, 1]]).cpu().numpy()
    out["test_total_error"] = float(np.abs(tpred - ty).mean())
    out["generalisation_gap"] = out["test_total_error"] - out["train_total_error"]

    if model.proj is not None:
        W = model.proj.weight.detach().cpu()
        sem = g.sem.cpu()
        gen = torch.Generator().manual_seed(0)
        a = torch.randint(0, len(sem), (n_pairs,), generator=gen)
        b = torch.randint(0, len(sem), (n_pairs,), generator=gen)
        keep = a != b
        din = (sem[a[keep]] - sem[b[keep]]).norm(dim=1)
        dout = ((sem[a[keep]] - sem[b[keep]]) @ W.T).norm(dim=1)
        ratio = (dout / din.clamp(min=1e-12)).numpy()
        out["lipschitz_projection"] = {"spectral_norm": float(torch.linalg.matrix_norm(W, 2)),
                                       "empirical_max": float(ratio.max()),
                                       "empirical_p95": float(np.quantile(ratio, 0.95))}
    return out


# --------------------------------------------------------------------------- latency vs d

def latency_vs_dimension(dims=(16, 32, 64, 128, 256, 512, 1024, 2048), n_entities: int = 50_000,
                         n_queries: int = 100_000, repeats: int = 5) -> list[dict]:
    """Per-query cosine cost on precomputed embeddings as d grows.

    Shows the O(d) dependence hidden by 'constant-time inference'.

    Args:
        dims: Embedding dimensions to time.
        n_entities: Number of random embeddings.
        n_queries: Queries per timing.
        repeats: Repetitions; the best time is kept.

    Returns:
        List of ``{"dim": d, "seconds_per_query": s}``.
    """
    rows = []
    for d in dims:
        z = torch.randn(n_entities, d)
        best = min(measure_query_seconds(z, n_queries, seed=r)["batched"] for r in range(repeats))
        rows.append({"dim": d, "seconds_per_query": best})
        print(f"  d={d:5d}  {best * 1e9:9.1f} ns/query")
    return rows


# --------------------------------------------------------------------------- statistics

def paired_tests(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict:
    """Paired t-tests per metric over matched runs, Benjamini-Hochberg corrected.

    Args:
        a: ``{metric: values over seeds}`` of the reference method.
        b: The same for the compared method.

    Returns:
        ``{metric: {"p_raw", "p_bh", "mean_diff"}}`` for metrics present in both.
    """
    names = [m for m in a if m in b and len(a[m]) == len(b[m]) and len(a[m]) > 1]
    raw = [float(stats.ttest_rel(a[m], b[m]).pvalue) for m in names]
    raw = [1.0 if np.isnan(p) else p for p in raw]
    adj = stats.false_discovery_control(raw, method="bh") if raw else []
    return {m: {"p_raw": p, "p_bh": float(q), "mean_diff": float(np.mean(np.subtract(a[m], b[m])))}
            for m, p, q in zip(names, raw, adj)}


def fmt_metrics(res: dict, keys=("mae", "rmse", "f1", "spearman")) -> str:
    """Format selected metrics on one line.

    Args:
        res: Metrics dict.
        keys: Metric names to include if present.

    Returns:
        ``"mae=... rmse=..."``.
    """
    return "  ".join(f"{k}={res[k]:.4f}" for k in keys if k in res)


def main() -> None:
    """Command-line entry point for evaluation."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    ap.add_argument("--labels", type=Path, help="for --baselines")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--semantic", default=DEFAULT_SEMANTIC)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    results = {}
    for ck in args.checkpoint:
        r = evaluate_checkpoint(ck, args.device)
        results[ck.stem] = r
        for which in ("uniform", "stratified"):
            print(f"{ck.stem}\n  test[{which}]  {fmt_metrics(r['test'][which])}")
        c = r["cost"]
        print(f"  offline {c['offline_total']:.1f}s, query {c['query_seconds']['batched'] * 1e6:.3f} us "
              f"(batched) / {c['query_seconds']['single'] * 1e6:.1f} us (single), exact ref "
              f"{c['exact_reference']['seconds']:.2f}s for {c['exact_reference']['pairs']} pairs, "
              f"break-even {c['break_even_queries']}")
    if args.baselines:
        results["baselines"] = baseline_scores(args.labels, args.semantic)
        for name, r in results["baselines"].items():
            print(f"{name:28s} uniform: {fmt_metrics(r['test']['uniform'])} | "
                      f"stratified: {fmt_metrics(r['test']['stratified'])}")
    if args.latency:
        results["latency"] = latency_vs_dimension()
    out = args.out or RESULTS_ROOT / "evaluate_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
