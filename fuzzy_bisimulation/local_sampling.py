"""Subgraph sampling and local exact references (Section IV-C, Section V-A).

Two kinds of subgraphs are used:

* Degree-aware training subgraphs: entities drawn with p(e) ∝ deg(e)^γ and the
  induced subgraph G_s = (E_s, R, T_s, w).  Algorithm 1 on G_s gives B*_s, the
  supervision target.  Dropping edges that leave E_s makes B*_s differ from
  B* (Assumption 1); experiments.py measures that gap.

* Local evaluation references, which avoid running Algorithm 1 on the whole
  graph (reviewer comment on the full-graph B* used for evaluation).
  B^(k)(s,t), the k-th iterate of Algorithm 1, only depends on transitions
  leaving states within distance k-1 of s or t.  So on the subgraph induced by
  the forward k-ball around a set of query entities, running k iterations
  gives the *full-graph* B^(k) for every pair of those entities, exactly.  If
  the BFS exhausts the forward closure before depth k, the ball contains every
  reachable state and running to convergence gives the full-graph B* exactly.
  The cost depends on the ball size, not on |E|.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .exact_bisimulation import TransitionSystem, compute_bisimulation, label_similarity


# --------------------------------------------------------------------------- entity sampling

def degree_aware_sample(degree: np.ndarray, size: int, gamma: float = 0.7, seed: int = 0,
                        exclude: np.ndarray | None = None) -> np.ndarray:
    """Sample entities without replacement with p(e) ∝ deg(e)^γ.

    γ = 0 is uniform sampling; γ = 1 is proportional to degree.  Entities with
    degree 0 are never drawn.

    Args:
        degree: Degree of every entity.
        size: Number of entities to draw (capped by the number of candidates).
        gamma: Sampling bias γ.
        seed: Random seed.
        exclude: Entity ids that must not be drawn.

    Returns:
        Sorted array of sampled entity ids.
    """
    rng = np.random.default_rng(seed)
    p = np.power(degree.astype(np.float64), gamma)
    p[degree == 0] = 0.0
    if exclude is not None and len(exclude):
        p[exclude] = 0.0
    size = min(size, int((p > 0).sum()))
    return np.sort(rng.choice(len(degree), size=size, replace=False, p=p / p.sum()))


def random_sample(degree: np.ndarray, size: int, seed: int = 0,
                  exclude: np.ndarray | None = None) -> np.ndarray:
    """Uniform sampling among entities with positive degree.

    Args:
        degree: Degree of every entity.
        size: Number of entities to draw.
        seed: Random seed.
        exclude: Entity ids that must not be drawn.

    Returns:
        Sorted array of sampled entity ids.
    """
    return degree_aware_sample(degree, size, gamma=0.0, seed=seed, exclude=exclude)


@dataclass
class EntitySplit:
    """Disjoint entity sets for held-out evaluation.

    Attributes:
        train: Training entities (supervision subgraph).
        val: Validation entities (early stopping).
        test: Test entities.
    """
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_entities(degree: np.ndarray, train_ratio: float = 0.1, n_val: int = 500,
                   n_test: int = 300, strategy: str = "degree", gamma: float = 0.7,
                   seed: int = 0) -> EntitySplit:
    """Disjoint train / validation / test entity sets (Table III protocol).

    Training entities follow the sampling strategy; validation and test entities
    are drawn uniformly from the remaining entities so the held-out evaluation is
    not biased towards hubs.

    Args:
        degree: Degree of every entity.
        train_ratio: Fraction of entities used for training.
        n_val: Number of validation entities.
        n_test: Number of test entities.
        strategy: ``"degree"`` (degree-aware) or ``"random"``.
        gamma: Sampling bias for the degree-aware strategy.
        seed: Random seed.

    Returns:
        An ``EntitySplit``.
    """
    n = len(degree)
    n_train = max(2, int(round(train_ratio * n)))
    g = gamma if strategy == "degree" else 0.0
    train = degree_aware_sample(degree, n_train, gamma=g, seed=seed)
    val = random_sample(degree, n_val, seed=seed + 1, exclude=train)
    test = random_sample(degree, n_test, seed=seed + 2, exclude=np.concatenate([train, val]))
    return EntitySplit(train=train, val=val, test=test)


def sample_pairs(n: int, n_pairs: int, seed: int = 0) -> np.ndarray:
    """Sample distinct unordered pairs (i < j) of local indices 0..n-1.

    Args:
        n: Number of items.
        n_pairs: Number of pairs; all pairs are returned if it exceeds n(n-1)/2.
        seed: Random seed.

    Returns:
        Int array of shape (k, 2).
    """
    rng = np.random.default_rng(seed)
    total = n * (n - 1) // 2
    if n_pairs >= total:
        i, j = np.triu_indices(n, 1)
        return np.stack([i, j], 1)
    flat = rng.choice(total, size=n_pairs, replace=False)
    # Invert the row-major index of the strict upper triangle.
    i = (n - 2 - np.floor(np.sqrt(-8 * flat + 4 * n * (n - 1) - 7) / 2.0 - 0.5)).astype(np.int64)
    j = (flat + i + 1 - total + (n - i) * ((n - i) - 1) // 2).astype(np.int64)
    return np.stack([i, j], 1)


# --------------------------------------------------------------------------- subgraphs

def induced_subgraph(triples: np.ndarray, weights: np.ndarray,
                     nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triples with both ends in ``nodes``, re-indexed to positions in ``nodes``.

    Args:
        triples: Int array (m, 3) of (head, relation, tail).
        weights: Degrees, shape (m,).
        nodes: Entity ids of the subgraph.

    Returns:
        ``(sub_triples, sub_weights)`` with local entity indices.
    """
    n_max = int(max(triples[:, [0, 2]].max(), nodes.max())) + 1
    local = np.full(n_max, -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    h, t = local[triples[:, 0]], local[triples[:, 2]]
    keep = (h >= 0) & (t >= 0)
    sub = np.stack([h[keep], triples[keep, 1], t[keep]], 1)
    return sub, weights[keep]


def forward_ball(num_entities: int, triples: np.ndarray, seeds: np.ndarray, depth: int,
                 max_nodes: int | None = None) -> tuple[np.ndarray, int, bool]:
    """Entities reachable from ``seeds`` in at most ``depth`` forward steps.

    If adding a level would exceed ``max_nodes``, the BFS stops at the previous
    level, so the returned depth is always complete.

    Args:
        num_entities: Number of entities in the graph.
        triples: Int array (m, 3) of (head, relation, tail).
        seeds: Start entities.
        depth: Maximum number of forward steps.
        max_nodes: Optional cap on the ball size.

    Returns:
        ``(nodes, reached_depth, closure_complete)`` where ``nodes`` lists the
        seeds first, and ``closure_complete`` is True when no further entity is
        reachable.
    """
    adj = sp.csr_matrix((np.ones(len(triples), dtype=bool), (triples[:, 0], triples[:, 2])),
                        shape=(num_entities, num_entities))
    seen = np.zeros(num_entities, dtype=bool)
    seen[seeds] = True
    order = [np.asarray(seeds)]
    frontier = np.asarray(seeds)
    reached = 0
    for _ in range(depth):
        nxt = np.unique(adj[frontier].indices)
        nxt = nxt[~seen[nxt]]
        if len(nxt) == 0:
            return np.concatenate(order), depth, True
        if max_nodes is not None and seen.sum() + len(nxt) > max_nodes:
            break
        seen[nxt] = True
        order.append(nxt)
        frontier = nxt
        reached += 1
    # Closure is complete iff the last level has no unseen successors.
    tail = np.unique(adj[frontier].indices) if len(frontier) else np.array([], dtype=np.int64)
    complete = bool(len(tail) == 0 or seen[tail].all())
    return np.concatenate(order), reached, complete


@dataclass
class LocalReference:
    """Exact reference values among a set of centre entities.

    Attributes:
        B: Reference matrix (k, k) among the centres.
        exact_fixpoint: True if it equals the full-graph B*; otherwise it equals
            the full-graph B^(depth).
        depth: Depth actually covered by the ball.
        ball_nodes: Number of entities in the ball.
        ball_triples: Number of triples in the ball.
        iterations: Iterations of Algorithm 1.
        seconds: Wall-clock time.
    """
    B: torch.Tensor           # (len(centers), len(centers)) reference among the centers
    exact_fixpoint: bool      # True: equals full-graph B*;  False: equals full-graph B^(depth)
    depth: int
    ball_nodes: int
    ball_triples: int
    iterations: int
    seconds: float


def local_reference(num_entities: int, triples: np.ndarray, weights: np.ndarray,
                    num_relations: int, labels: np.ndarray | None, centers: np.ndarray,
                    depth: int = 3, max_nodes: int | None = 20000, eps: float = 1e-4,
                    max_iter: int | None = None, device: str = "cpu", semantics: str = "godel",
                    discount: float = 1.0) -> LocalReference:
    """Exact full-graph B^(depth) (or B*) among ``centers`` from a local computation.

    Runs Algorithm 1 on the subgraph induced by the forward ``depth``-ball around
    the centres.  ``labels`` are the full-graph propositions so that boundary
    states keep their true labels.

    Args:
        num_entities: Number of entities in the graph.
        triples: Int array (m, 3) of (head, relation, tail).
        weights: Degrees, shape (m,).
        num_relations: Number of relation types.
        labels: Full-graph proposition degrees (n, P), or None.
        centers: Entities whose pairwise values are needed.
        depth: k of B^(k).
        max_nodes: Optional cap on the ball size; the depth shrinks if exceeded.
        eps: Stopping tolerance when the closure is complete.
        max_iter: Optional iteration cap when the closure is complete.
        device: Torch device.
        semantics: Key of ``SEMANTICS``.
        discount: λ in (0, 1].

    Returns:
        A ``LocalReference``.
    """
    t0 = time.perf_counter()
    nodes, reached, complete = forward_ball(num_entities, triples, centers, depth, max_nodes)
    sub_t, sub_w = induced_subgraph(triples, weights, nodes)
    ts = TransitionSystem.from_triples(len(nodes), sub_t, sub_w, num_relations, device=device)
    L = label_similarity(None if labels is None else labels[nodes], len(nodes), device)
    kw = dict(semantics=semantics, discount=discount)
    if complete:
        res = compute_bisimulation(ts, L, eps=eps, max_iter=max_iter, **kw)
    else:
        res = compute_bisimulation(ts, L, eps=0.0, max_iter=max(reached, 1), **kw)
    k = len(centers)
    # Stabilising on the ball before ``depth`` iterations does not imply the
    # full-graph iteration has stabilised, so only a complete closure is B*.
    return LocalReference(B=res.B[:k, :k].cpu(), exact_fixpoint=complete,
                          depth=reached, ball_nodes=len(nodes), ball_triples=len(sub_t),
                          iterations=res.iterations, seconds=time.perf_counter() - t0)


def induced_reference(triples: np.ndarray, weights: np.ndarray, num_relations: int,
                      labels: np.ndarray | None, nodes: np.ndarray, eps: float = 1e-4,
                      max_iter: int | None = None, device: str = "cpu", semantics: str = "godel",
                      discount: float = 1.0):
    """Algorithm 1 on the subgraph induced by ``nodes`` (the paper's B*_s).

    Args:
        triples: Int array (m, 3) of (head, relation, tail).
        weights: Degrees, shape (m,).
        num_relations: Number of relation types.
        labels: Full-graph proposition degrees (n, P), or None.
        nodes: Entities of the subgraph.
        eps: Stopping tolerance.
        max_iter: Optional iteration cap.
        device: Torch device.
        semantics: Key of ``SEMANTICS``.
        discount: λ in (0, 1].

    Returns:
        ``(result, num_subgraph_triples)`` where ``result`` is a
        ``BisimulationResult`` over the local indices of ``nodes``.
    """
    t0 = time.perf_counter()
    sub_t, sub_w = induced_subgraph(triples, weights, nodes)
    ts = TransitionSystem.from_triples(len(nodes), sub_t, sub_w, num_relations, device=device)
    L = label_similarity(None if labels is None else labels[nodes], len(nodes), device)
    res = compute_bisimulation(ts, L, eps=eps, max_iter=max_iter, semantics=semantics,
                               discount=discount)
    res.seconds = time.perf_counter() - t0
    return res, len(sub_t)
