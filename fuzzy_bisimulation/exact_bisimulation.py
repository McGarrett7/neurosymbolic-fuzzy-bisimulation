"""Exact fuzzy bisimulation under Gödel semantics (Section III, Algorithm 1).

F(R)(s,t) = min( L(s,t),
                 min_a min_s' [ δ(s,a,s') ⇒ max_t' min(δ(t,a,t'), R(s',t')) ],
                 min_a min_t' [ δ(t,a,t') ⇒ max_s' min(δ(s,a,s'), R(s',t')) ] )

B* = νR.F(R) is reached by iterating F from R ≡ 1 (Knaster–Tarski; Gödel
degrees form a finite set, so the iteration terminates).

Gödel semantics with discount 1 is the paper's definition and the default.
Two optional generalisations exist because, on real graphs, Gödel B* is
almost binary (x ⇒ 0 = 0, so a single unmatched transition at any depth
zeroes the degree); they keep F monotone, so the greatest fixpoint exists:

* ``semantics``: the residuated pair (⊗, ⇒) can be Łukasiewicz
  (max(0, x+y-1), min(1, 1-x+y)) or product (x·y, 1 if x <= y else y/x).
* ``discount`` λ ∈ (0, 1]:  F_λ(R) = min(L, (1-λ) + λ·min(fwd, bwd)), so a
  mismatch found k steps away costs about λ^k (as in discounted bisimulation
  metrics).  λ = 1 recovers Eq. (1).

Implementation: instead of the dense O(|R| n^3) max-min products, every
iteration loops over transitions only.  For relation a with edge set E_a,

    M_a[s', t] = max_{(t,a,t') ∈ E_a} min(w, R[s', t'])      (rows s' ∈ targets of a)
    fwd[s, t]  = min_{(s,a,s') ∈ E_a} ( w ⇒ M_a[s', t] )

costs O(|E_a| · n), i.e. O(|T| · n) per iteration overall with O(n^2) memory.
Absent transitions impose no constraint (δ = 0 ⇒ x = 1); a state with an
a-move against one with none gets w ⇒ 0 = 0.
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_CHUNK = 1 << 24   # elements per temporary block (64 MB in float32)


# --------------------------------------------------------------------------- Gödel algebra

def godel_tnorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, y)


def godel_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x ⇒ y = 1 if x <= y else y."""
    return torch.where(x <= y, torch.ones_like(y), y)


def godel_equiv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x ⇔ y = (x ⇒ y) ⊗ (y ⇒ x) = 1 if x == y else min(x, y)."""
    return torch.minimum(godel_impl(x, y), godel_impl(y, x))


def lukasiewicz_tnorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x + y - 1.0).clamp(min=0.0)


def lukasiewicz_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (1.0 - x + y).clamp(max=1.0)


def product_tnorm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def product_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= y, torch.ones_like(y), y / x.clamp(min=1e-12))


SEMANTICS = {
    "godel": (godel_tnorm, godel_impl),
    "lukasiewicz": (lukasiewicz_tnorm, lukasiewicz_impl),
    "product": (product_tnorm, product_impl),
}


# --------------------------------------------------------------------------- transition system

@dataclass
class TransitionSystem:
    """FLTS view of a fuzzy KG: entities = states, relations = actions."""
    n: int
    num_relations: int
    rel_edges: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(default_factory=list)

    @classmethod
    def from_triples(cls, n: int, triples: np.ndarray, weights: np.ndarray, num_relations: int,
                     device: str | torch.device = "cpu") -> "TransitionSystem":
        ts = cls(n=n, num_relations=num_relations)
        triples = np.asarray(triples)
        weights = np.asarray(weights, dtype=np.float32)
        keep = weights > 0                      # δ = 0 is the same as no transition
        triples, weights = triples[keep], weights[keep]
        order = np.argsort(triples[:, 1], kind="stable")
        triples, weights = triples[order], weights[order]
        bounds = np.searchsorted(triples[:, 1], np.arange(num_relations + 1))
        for r in range(num_relations):
            lo, hi = bounds[r], bounds[r + 1]
            if hi > lo:
                ts.rel_edges.append((
                    torch.as_tensor(triples[lo:hi, 0], device=device),
                    torch.as_tensor(triples[lo:hi, 2], device=device),
                    torch.as_tensor(weights[lo:hi], device=device),
                ))
        return ts

    @property
    def num_transitions(self) -> int:
        return sum(len(s) for s, _, _ in self.rel_edges)


# --------------------------------------------------------------------------- state labels

def build_state_labels(n: int, triples: np.ndarray, weights: np.ndarray, num_relations: int,
                       mode: str = "inout") -> np.ndarray | None:
    """Fuzzy propositions L(s, p) derived from the graph.

    none   no propositions, L(s,t) = 1
    out    p_r^out(s) = max_t w(s, r, t)
    in     p_r^in(s)  = max_h w(h, r, s)
    inout  both (default): without incoming propositions every sink entity
           would be bisimilar to every other sink, which is uninformative for
           KGs where many entities only appear as tails.
    """
    if mode == "none":
        return None
    triples = np.asarray(triples)
    blocks = []
    if mode in ("out", "inout"):
        lab = np.zeros((n, num_relations), dtype=np.float32)
        np.maximum.at(lab, (triples[:, 0], triples[:, 1]), weights)
        blocks.append(lab)
    if mode in ("in", "inout"):
        lab = np.zeros((n, num_relations), dtype=np.float32)
        np.maximum.at(lab, (triples[:, 2], triples[:, 1]), weights)
        blocks.append(lab)
    if not blocks:
        raise ValueError(f"unknown label mode {mode!r}")
    return np.concatenate(blocks, axis=1)


def label_similarity(labels: np.ndarray | torch.Tensor | None, n: int,
                     device: str | torch.device = "cpu") -> torch.Tensor | None:
    """L(s,t) = 1/|P| Σ_p (L(s,p) ⇔ L(t,p)).

    Propositions are sparse, so the sum splits into pairs where both degrees
    are 0 (contribute 1; counted with one matrix product) and pairs where both
    are positive (handled per proposition).  Pairs with exactly one zero
    contribute min(x, 0) = 0.
    """
    if labels is None:
        return None
    X = torch.as_tensor(labels, dtype=torch.float32, device=device)
    P = X.shape[1]
    nz = (X > 0).float()
    cnt = nz.sum(1)
    L = nz @ nz.T                                          # |supp(s) ∩ supp(t)|
    L.add_(P).sub_(cnt[:, None]).sub_(cnt[None, :])        # #props where both are 0
    for p in range(P):
        idx = torch.nonzero(X[:, p] > 0).squeeze(1)
        if len(idx) == 0:
            continue
        v = X[idx, p]
        step = max(1, DEFAULT_CHUNK // len(idx))
        for a in range(0, len(idx), step):
            rows = idx[a:a + step]
            L[rows[:, None], idx[None, :]] += godel_equiv(v[a:a + step, None], v[None, :])
    return L.div_(P).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- operator

def _forward_half(R: torch.Tensor, ts: TransitionSystem, chunk: int,
                  semantics: str = "godel") -> torch.Tensor:
    """fwd[s,t] = min_a min_{(s,a,s')} δ ⇒ max_{(t,a,t')} δ ⊗ R[s',t']."""
    tnorm, impl = SEMANTICS[semantics]
    n = ts.n
    out = torch.ones(n, n, dtype=R.dtype, device=R.device)
    for src, dst, w in ts.rel_edges:
        targets, dst_row = torch.unique(dst, return_inverse=True)
        M = torch.zeros(len(targets), n, dtype=R.dtype, device=R.device)
        step = max(1, chunk // len(targets))
        for a in range(0, len(src), step):
            s_, d_, w_ = src[a:a + step], dst[a:a + step], w[a:a + step]
            vals = tnorm(w_[None, :].expand(len(targets), -1), R[targets[:, None], d_[None, :]])
            M.scatter_reduce_(1, s_[None, :].expand(len(targets), -1), vals, reduce="amax")
        step = max(1, chunk // n)
        for a in range(0, len(src), step):
            s_, w_ = src[a:a + step], w[a:a + step]
            vals = impl(w_[:, None].expand(-1, n), M[dst_row[a:a + step]])
            out.scatter_reduce_(0, s_[:, None].expand(-1, n), vals, reduce="amin")
    return out


def bisimulation_operator(R: torch.Tensor, ts: TransitionSystem, L: torch.Tensor | None = None,
                          symmetric: bool = False, chunk: int = DEFAULT_CHUNK,
                          semantics: str = "godel", discount: float = 1.0) -> torch.Tensor:
    """One application of F (Eq. 1).  ``symmetric=True`` skips the second pass
    when R is known to be symmetric (then the backward term is fwd^T)."""
    fwd = _forward_half(R, ts, chunk, semantics)
    bwd = fwd.T if symmetric else _forward_half(R.T.contiguous(), ts, chunk, semantics).T
    out = torch.minimum(fwd, bwd)
    if discount != 1.0:
        out = out.mul_(discount).add_(1.0 - discount)
    if L is not None:
        out = torch.minimum(out, L)
    return out


@dataclass
class BisimulationResult:
    B: torch.Tensor
    iterations: int
    converged: bool
    deltas: list[float]
    seconds: float


def compute_bisimulation(ts: TransitionSystem, L: torch.Tensor | None = None, eps: float = 1e-4,
                         max_iter: int | None = None, chunk: int = DEFAULT_CHUNK, semantics: str = "godel",
                         discount: float = 1.0, verbose: bool = False) -> BisimulationResult:
    """Algorithm 1: iterate F from B ≡ 1 until max |B(k+1) - B(k)| < eps.

    ``max_iter = None`` (default) runs until the stopping rule holds, as in the
    paper; the iteration is monotone and bounded, so it terminates.  With ``max_iter = k`` and ``eps = 0`` this returns the depth-k
    approximation B^(k) ≥ B*, which local_sampling uses for exact local
    references.
    """
    t0 = time.perf_counter()
    device = L.device if L is not None else (ts.rel_edges[0][0].device if ts.rel_edges else "cpu")
    B = torch.ones(ts.n, ts.n, dtype=torch.float32, device=device)
    symmetric = L is None or bool(torch.equal(L, L.T))
    deltas: list[float] = []
    converged = False
    for k in (itertools.count() if max_iter is None else range(max_iter)):
        B_next = bisimulation_operator(B, ts, L, symmetric=symmetric, chunk=chunk,
                                       semantics=semantics, discount=discount)
        delta = float((B_next - B).abs().max()) if ts.n else 0.0
        deltas.append(delta)
        B = B_next
        if verbose:
            print(f"    iter {k + 1}: delta={delta:.6f}")
        if delta < eps or (eps == 0 and delta == 0):
            converged = True
            break
    return BisimulationResult(B=B, iterations=len(deltas), converged=converged, deltas=deltas,
                              seconds=time.perf_counter() - t0)


def bisimulation_from_triples(n: int, triples: np.ndarray, weights: np.ndarray, num_relations: int,
                              label_mode: str = "inout", eps: float = 1e-4, max_iter: int | None = None,
                              device: str = "cpu", semantics: str = "godel", discount: float = 1.0,
                              verbose: bool = False) -> BisimulationResult:
    """Convenience wrapper: labels + transition system + Algorithm 1."""
    t0 = time.perf_counter()
    ts = TransitionSystem.from_triples(n, triples, weights, num_relations, device=device)
    L = label_similarity(build_state_labels(n, triples, weights, num_relations, label_mode), n, device)
    res = compute_bisimulation(ts, L, eps=eps, max_iter=max_iter, semantics=semantics,
                               discount=discount, verbose=verbose)
    res.seconds = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------- reference check

def naive_operator(R: np.ndarray, delta: np.ndarray, L: np.ndarray | None,
                   semantics: str = "godel", discount: float = 1.0) -> np.ndarray:
    """Literal transcription of Eq. (1); delta has shape (|Act|, n, n)."""
    A, n, _ = delta.shape
    tn, im = SEMANTICS[semantics]
    tnorm = lambda x, y: float(tn(torch.tensor(float(x)), torch.tensor(float(y))))
    imp = lambda x, y: float(im(torch.tensor(float(x)), torch.tensor(float(y))))
    out = np.ones((n, n))
    for s in range(n):
        for t in range(n):
            v = 1.0
            for a in range(A):
                for s2 in range(n):
                    best = max(tnorm(delta[a, t, t2], R[s2, t2]) for t2 in range(n))
                    v = min(v, imp(delta[a, s, s2], best))
                for t2 in range(n):
                    best = max(tnorm(delta[a, s, s2], R[s2, t2]) for s2 in range(n))
                    v = min(v, imp(delta[a, t, t2], best))
            v = (1.0 - discount) + discount * v
            out[s, t] = v if L is None else min(v, L[s, t])
    return out


def selftest(trials: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        n, A = int(rng.integers(3, 8)), int(rng.integers(1, 4))
        m = int(rng.integers(n, 3 * n))
        triples = np.stack([rng.integers(0, n, m), rng.integers(0, A, m), rng.integers(0, n, m)], 1)
        triples = np.unique(triples, axis=0)
        weights = np.round(rng.choice([0.2, 0.5, 0.7, 1.0], len(triples)), 3).astype(np.float32)
        delta = np.zeros((A, n, n))
        delta[triples[:, 1], triples[:, 0], triples[:, 2]] = weights
        labels = build_state_labels(n, triples, weights, A, "inout")
        Lt = label_similarity(labels, n)
        Ln = np.array([[np.mean([1.0 if x == y else min(x, y) for x, y in zip(labels[s], labels[t])])
                        for t in range(n)] for s in range(n)])
        assert np.allclose(Lt.numpy(), Ln, atol=1e-6), "label similarity mismatch"

        ts = TransitionSystem.from_triples(n, triples, weights, A)
        R = rng.choice([0.0, 0.3, 0.6, 1.0], size=(n, n)).astype(np.float32)   # not symmetric
        sem = list(SEMANTICS)[trial % len(SEMANTICS)]
        kw = dict(semantics=sem, discount=1.0 if trial % 2 == 0 else 0.8)
        fast = bisimulation_operator(torch.as_tensor(R), ts, Lt, symmetric=False, chunk=7, **kw).numpy()
        slow = naive_operator(R, delta, Ln, **kw)
        assert np.allclose(fast, slow, atol=1e-5), f"operator mismatch in trial {trial} ({kw})"

        res = compute_bisimulation(ts, Lt, eps=1e-7 if sem != "godel" else 0.0, max_iter=2000, **kw)
        B = res.B.numpy()
        assert res.converged, kw
        assert np.allclose(naive_operator(B, delta, Ln, **kw), B, atol=1e-5), f"not a fixpoint {kw}"
        assert np.allclose(B, B.T) and np.allclose(np.diag(B), 1.0)
    print(f"selftest passed ({trials} random FLTSs, all semantics, discount 1 / 0.8): "
          f"vectorised operator == Eq. (1), "
          f"result is a symmetric reflexive fixpoint")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact fuzzy bisimulation (Algorithm 1).")
    ap.add_argument("--selftest", action="store_true", help="check against a literal Eq. (1)")
    ap.add_argument("--dataset", help="time Algorithm 1 on a degree-aware sample of this KG")
    ap.add_argument("--entities", type=int, default=1000)
    ap.add_argument("--confidence", default="auto")
    ap.add_argument("--label-mode", default="inout", choices=["none", "out", "in", "inout"])
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--semantics", default="godel", choices=list(SEMANTICS))
    ap.add_argument("--discount", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.selftest or not args.dataset:
        selftest()
    if args.dataset:
        from .fuzzy_kg import load_kg
        from .local_sampling import degree_aware_sample, induced_subgraph
        kg = load_kg(args.dataset, confidence=args.confidence)
        print(kg.summary())
        nodes = degree_aware_sample(kg.degree(), args.entities, gamma=0.7, seed=0)
        sub_t, sub_w = induced_subgraph(kg.triples, kg.weights, nodes)
        res = bisimulation_from_triples(len(nodes), sub_t, sub_w, kg.num_relations, args.label_mode,
                                        args.eps, device=args.device, semantics=args.semantics,
                                        discount=args.discount, verbose=True)
        vals = res.B[torch.triu_indices(len(nodes), len(nodes), 1).unbind()]
        print(f"n_s={len(nodes)} |T_s|={len(sub_t)} iterations={res.iterations} "
              f"converged={res.converged} time={res.seconds:.2f}s "
              f"mean B={vals.mean():.3f} frac(B>=0.7)={(vals >= 0.7).float().mean():.3f}")


if __name__ == "__main__":
    main()
