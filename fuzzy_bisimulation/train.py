"""Stages 2-3: learning bisimulation-preserving embeddings (Section IV-D/E, Algorithm 2).

    python -m fuzzy_bisimulation.train --labels data/<dataset>/labels/<tag>.pt

Model:  B̂(e_i, e_j) = (cos(z_i, z_j) + 1) / 2,   z = α·g(e) + (1-α)·ϕ_proj(e)

Defaults follow the paper (Algorithm 2, Section IV-D/E, Table IV):

* g: two-layer R-GCN, d = 128, h' = ReLU(W_0 h_i + Σ_r Σ_{j∈N_r(i)} W_r h_j / |N_r(i)|)
  with a full W_r per KG relation.  Messages flow from successors to an
  entity (bisimulation of s is decided by the moves s -> s').
* Input features h^(0) (not specified in the paper) are inductive
  per-relation in/out weight statistics, so held-out entities get
  embeddings without having been trained on.
* ϕ_proj(e) = W_proj ϕ(e) + b_proj, ϕ = OpenAI text-embedding-3-large (768) from semantic.py.
* z = α·g + (1-α)·ϕ_proj with α = 0.7, no normalisation before fusion.
* Each epoch shuffles the training pair set P and runs over all its batches;
  MSE loss, Adam, lr decay 0.95 / 10 epochs, early stopping on validation
  loss with patience 10.  P is every pair of training entities unless
  --train-pairs caps it.

Optional deviations (off by default): --bases K (basis-decomposed W_r),
--inverse (reverse edges as extra relations), --edge-weight (messages scaled
by the fuzzy degree), --normalize-fusion (L2-normalise both channels so α is
not confounded with their norms), --steps-per-epoch N (N random
batches per epoch instead of a pass over P, needed for --pair-sampling balanced).

Deviation on by default: no ReLU after the last GNN layer.  The paper applies
σ = ReLU on every layer; then g >= 0, cos >= 0 and B̂ >= 0.5 for every α = 1
model (R-GCN/GCN/SAGE/MLP baselines, the w/o-semantic ablation), which can
never predict low similarities and makes the baseline comparison unfair.
--final-relu restores the paper's configuration (used as an ablation).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import CHECKPOINT_ROOT, DATA_ROOT
from .fuzzy_kg import FuzzyKG, load_kg
from .local_sampling import sample_pairs
from .model import ENCODERS, BisimNet, GraphTensors, build_graph, structural_features
from .semantic import load_or_encode

PAPER_SEMANTIC = "openai:text-embedding-3-large:768"           # Table IV
DEFAULT_SEMANTIC = "st:all-mpnet-base-v2"                      # open 768-d stand-in, no API key


# --------------------------------------------------------------------------- data for training

def kg_for_labels(payload: dict) -> FuzzyKG:
    """Reload the knowledge graph a label file was generated from.

    Args:
        payload: Loaded label file.

    Returns:
        The ``FuzzyKG`` with the same dataset, confidence scheme and seed.
    """
    cfg = payload["config"]
    return load_kg(cfg["dataset"], Path(cfg["data_root"]), confidence=cfg["confidence"], seed=cfg["seed"])


def eval_pairs(part: dict, which: str = "both") -> tuple[np.ndarray, np.ndarray]:
    """Global entity pairs and reference values of a val/test block.

    Args:
        part: The ``"val"`` or ``"test"`` block of a label file.
        which: ``"uniform"``, ``"stratified"`` or ``"both"``.

    Returns:
        ``(pairs, values)``: global entity-id pairs (p, 2) and their references (p,).
    """
    keys = {"uniform": ["pairs_uniform"], "stratified": ["pairs_stratified"],
            "both": ["pairs_uniform", "pairs_stratified"]}[which]
    pairs = np.concatenate([part[k] for k in keys])
    B = part["B"].float().numpy()
    return part["entities"][pairs], B[pairs[:, 0], pairs[:, 1]]


def jaccard_targets(kg: FuzzyKG, entities: np.ndarray) -> torch.Tensor:
    """Neighbourhood Jaccard similarity, the 'w/o symbolic supervision' target.

    Keys are (direction, relation, neighbour) triples.

    Args:
        kg: The knowledge graph.
        entities: Training entities.

    Returns:
        Float tensor (k, k) of Jaccard similarities among ``entities``.
    """
    import scipy.sparse as sp
    n, R = kg.num_entities, kg.num_relations
    h, r, t = kg.triples.T
    rows = np.concatenate([h, t])
    cols = np.concatenate([r * n + t, (R + r) * n + h])
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, 2 * R * n))[entities]
    X.data[:] = 1.0
    inter = (X @ X.T).toarray()
    size = np.asarray(X.sum(1)).ravel()
    union = size[:, None] + size[None, :] - inter
    return torch.as_tensor(np.where(union > 0, inter / np.maximum(union, 1), 1.0), dtype=torch.float32)


class PairSampler:
    """Random training pairs drawn from the training-entity matrix."""
    def __init__(self, B: torch.Tensor, mode: str, tau: float, seed: int):
        """Create the sampler.

        Args:
            B: Target matrix among training entities (k, k).
            mode: ``"uniform"``, or ``"balanced"`` to draw half of every batch from
                pairs with target >= ``tau``.
            tau: Positive threshold for balanced sampling.
            seed: Random seed.
        """
        self.B, self.mode, self.n = B, mode, B.shape[0]
        self.gen = torch.Generator().manual_seed(seed)
        if mode == "balanced":
            pos = torch.nonzero(torch.triu(B >= tau, 1))
            self.pos = pos if len(pos) else None

    def sample(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a batch of pairs with i != j.

        Args:
            k: Batch size.

        Returns:
            ``(i, j)`` local index tensors of shape (k,).
        """
        i = torch.randint(0, self.n, (k,), generator=self.gen)
        j = (i + torch.randint(1, self.n, (k,), generator=self.gen)) % self.n     # j != i
        if self.mode == "balanced" and self.pos is not None:
            half = k // 2
            pick = self.pos[torch.randint(0, len(self.pos), (half,), generator=self.gen)]
            i[:half], j[:half] = pick[:, 0], pick[:, 1]
        return i, j


# --------------------------------------------------------------------------- training

def add_train_args(ap: argparse.ArgumentParser) -> None:
    """Register the training command-line options.

    Args:
        ap: Parser to extend.
    """
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--encoder", default="rgcn", choices=ENCODERS)
    ap.add_argument("--semantic", default=DEFAULT_SEMANTIC,
                    help="semantic encoder spec (see semantic.py) or 'none'; paper: " + PAPER_SEMANTIC)
    ap.add_argument("--alpha", default="0.7", help="fusion weight in [0,1] or 'learn'")
    ap.add_argument("--normalize-fusion", action="store_true", help="L2-normalise g and ϕ_proj before fusion")
    ap.add_argument("--final-relu", action="store_true",
                    help="paper's ReLU after the last GNN layer (bounds B_hat >= 0.5 when alpha = 1)")
    ap.add_argument("--calibrate-output", action="store_true",
                    help="B_hat = sigmoid(a*cos + b) with learned a, b instead of (cos + 1) / 2")
    ap.add_argument("--target", default="bisim", choices=["bisim", "jaccard"])
    ap.add_argument("--pair-sampling", default="uniform", choices=["uniform", "balanced"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bases", type=int, default=0, help="0: full W_r per relation (paper); K: K bases")
    ap.add_argument("--inverse", action="store_true", help="also pass messages along reversed edges")
    ap.add_argument("--edge-weight", action="store_true", help="scale messages by the fuzzy degree")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="0: one shuffled pass over P per epoch (paper); N: N random batches")
    ap.add_argument("--train-pairs", type=int, default=0, help="|P|; 0: all training-entity pairs")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")


def model_kwargs(a: argparse.Namespace, g: GraphTensors) -> dict:
    """Constructor arguments of ``BisimNet`` for the given options.

    Args:
        a: Parsed training options.
        g: Graph tensors (used for feature, relation and semantic dimensions).

    Returns:
        Keyword arguments for ``BisimNet``.
    """
    alpha = a.alpha if a.alpha == "learn" else float(a.alpha)
    return dict(encoder=a.encoder, feat_dim=g.x.shape[1], num_relations=g.num_relations,
                sem_dim=None if g.sem is None else g.sem.shape[1], d=a.dim, layers=a.layers,
                num_bases=a.bases, dropout=a.dropout, alpha=alpha,
                normalize_fusion=a.normalize_fusion, final_relu=a.final_relu,
                calibrate_output=getattr(a, "calibrate_output", False))


def deviations(a: argparse.Namespace) -> str:
    """Short tag of the options that depart from the paper.

    Args:
        a: Parsed training options.

    Returns:
        Dash-separated flags, or ``"paper"`` if none.
    """
    flags = [f"b{a.bases}" if a.bases else "", "inv" if a.inverse else "", "ew" if a.edge_weight else "",
             "nf" if a.normalize_fusion else "", "" if a.final_relu else "nofr",
             "cal" if getattr(a, "calibrate_output", False) else "",
             f"spe{a.steps_per_epoch}" if a.steps_per_epoch else "",
             f"P{a.train_pairs}" if a.train_pairs else ""]
    return "-".join(f for f in flags if f) or "paper"


def checkpoint_name(a: argparse.Namespace) -> str:
    """Checkpoint file name for the given options.

    Args:
        a: Parsed training options.

    Returns:
        Name encoding dataset, label tag, encoder, semantic model, α, target,
        pair sampling, deviations and seed.
    """
    sem = a.semantic.replace(":", "-").replace("/", "-")
    return (f"{a.labels.parent.parent.name}__{a.labels.stem}__{a.encoder}_{sem}_a{a.alpha}"
            f"_{a.target}_{a.pair_sampling}_{deviations(a)}_s{a.seed}")


def train(a: argparse.Namespace) -> Path:
    """Stages 2-3: train the approximator with Algorithm 2.

    MSE to the reference values, Adam with lr decay 0.95 every 10 epochs, and
    early stopping on validation loss.  The best state is saved.

    Args:
        a: Parsed training options.

    Returns:
        Path of the saved checkpoint.

    Raises:
        ValueError: If balanced pair sampling is used without ``--steps-per-epoch``.
    """
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    payload = torch.load(a.labels, weights_only=False)
    kg = kg_for_labels(payload)
    sem, sem_seconds = load_or_encode(kg, a.semantic, Path(payload["config"]["data_root"]))
    g = build_graph(kg, sem, inverse=a.inverse, edge_weight=a.edge_weight).to(a.device)
    model = BisimNet(**model_kwargs(a, g)).to(a.device)

    train_ent = torch.as_tensor(payload["train"]["entities"], device=a.device)
    if a.target == "bisim":
        B_train = payload["train"]["B"].float()
    else:
        B_train = jaccard_targets(kg, payload["train"]["entities"])
    if a.steps_per_epoch > 0:
        sampler = PairSampler(B_train, a.pair_sampling, a.tau, a.seed)

        def batches():
            """Yield ``steps_per_epoch`` random batches of training pairs.

            Yields:
                ``(i, j)`` local index tensors.
            """
            for _ in range(a.steps_per_epoch):
                yield sampler.sample(a.batch_size)
    else:
        if a.pair_sampling != "uniform":
            raise ValueError("--pair-sampling balanced needs --steps-per-epoch > 0")
        n_s = B_train.shape[0]
        total = n_s * (n_s - 1) // 2
        P = torch.as_tensor(sample_pairs(n_s, a.train_pairs if a.train_pairs > 0 else total, a.seed))
        gen = torch.Generator().manual_seed(a.seed)

        def batches():
            """Yield one shuffled pass over the training pair set P (Algorithm 2, line 5).

            Yields:
                ``(i, j)`` local index tensors.
            """
            perm = torch.randperm(len(P), generator=gen)              # Algorithm 2, line 5
            for k in range(0, len(P), a.batch_size):
                b = P[perm[k:k + a.batch_size]]
                yield b[:, 0], b[:, 1]
    B_train = B_train.to(a.device)
    vp, vy = eval_pairs(payload["val"])
    vp, vy = torch.as_tensor(vp, device=a.device), torch.as_tensor(vy, device=a.device)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
    best, best_state, bad, history = float("inf"), None, 0, []
    t_start = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        losses = []
        for i, j in batches():
            i, j = i.to(a.device), j.to(a.device)
            z = model.embed(g)
            pred = model.similarity(z[train_ent[i]], z[train_ent[j]])
            loss = F.mse_loss(pred, B_train[i, j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            z = model.embed(g)
            vpred = model.similarity(z[vp[:, 0]], z[vp[:, 1]])
            vloss = F.mse_loss(vpred, vy).item()
            vmae = (vpred - vy).abs().mean().item()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vloss,
                        "val_mae": vmae, "alpha": float(model.alpha)})
        if not a.quiet:
            print(f"epoch {epoch:3d} train {np.mean(losses):.4f} val {vloss:.4f} mae {vmae:.4f} "
                  f"alpha {float(model.alpha):.3f}")
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                if not a.quiet:
                    print(f"early stopping at epoch {epoch}")
                break
    train_seconds = time.perf_counter() - t_start

    out = a.out or CHECKPOINT_ROOT / f"{checkpoint_name(a)}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    torch.save({"state_dict": best_state, "model_kwargs": model_kwargs(a, g), "config": cfg,
                "labels": str(a.labels), "history": history, "best_val_loss": best,
                "timing": {"train": train_seconds, "semantic_embedding": sem_seconds,
                           "epochs": len(history)}}, out)
    print(f"saved {out}  (best val MSE {best:.4f}, {len(history)} epochs, {train_seconds:.1f}s)")
    return out


def load_model(ckpt_path: Path, device: str = "cpu", weights: np.ndarray | None = None
               ) -> tuple[BisimNet, GraphTensors, dict, dict]:
    """Rebuild the model and graph tensors from a checkpoint.

    Args:
        ckpt_path: Checkpoint file.
        device: Torch device.
        weights: Optional degrees overriding the KG's (perturbation experiment).

    Returns:
        ``(model, graph, checkpoint, label_payload)``; the model is in eval mode.
    """
    ckpt = torch.load(ckpt_path, weights_only=False)
    payload = torch.load(ckpt["labels"], weights_only=False)
    cfg = ckpt["config"]
    kg = kg_for_labels(payload)
    sem, _ = load_or_encode(kg, cfg["semantic"], Path(payload["config"]["data_root"]))
    # Checkpoints from before the paper-default switch stored no_inverse / no_edge_weight
    # and had no ReLU on the last layer.
    inverse = cfg["inverse"] if "inverse" in cfg else not cfg["no_inverse"]
    edge_weight = cfg["edge_weight"] if "edge_weight" in cfg else not cfg["no_edge_weight"]
    g = build_graph(kg, sem, inverse=inverse, edge_weight=edge_weight, weights=weights).to(device)
    model = BisimNet(**{"final_relu": False, **ckpt["model_kwargs"]}).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, g, ckpt, payload


def main() -> None:
    """Command-line entry point for training."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_train_args(ap)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
