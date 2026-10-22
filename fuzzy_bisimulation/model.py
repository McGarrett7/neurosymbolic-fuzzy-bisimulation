"""Graph tensors and the bisimulation network (Section IV-D/E).

B̂(e_i, e_j) = (cos(z_i, z_j) + 1) / 2,   z = α·g(e) + (1-α)·ϕ_proj(e)

Defaults follow the paper; see train.py for the optional deviations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fuzzy_kg import FuzzyKG

ENCODERS = ["rgcn", "gcn", "sage", "mlp"]


# --------------------------------------------------------------------------- graph tensors

@dataclass
class GraphTensors:
    """Tensors describing the message-passing graph.

    Attributes:
        x: Structural input features (n, F).
        src: Message source of every edge (E,).
        dst: Message receiver of every edge (E,).
        rel: Relation id of every edge, including inverse offsets (E,).
        norm: 1 / |N_r(dst)|, or w / |N_r(dst)| with edge weights (E,).
        weight: Raw fuzzy degree of every edge (E,).
        num_relations: Number of edge relation types.
        sem: Frozen semantic embeddings (n, D), or None.
        key_index: Edge -> its (relation, receiver) group (E,).
        key_dst: Receiver of each group; groups are sorted by relation (K,).
        rel_ptr: Groups of relation r are ``key_dst[rel_ptr[r]:rel_ptr[r + 1]]``.
    """
    x: torch.Tensor            # (n, F) structural input features
    src: torch.Tensor          # message source  (E,)
    dst: torch.Tensor          # message receiver (E,)
    rel: torch.Tensor          # relation id incl. inverse offsets (E,)
    norm: torch.Tensor         # 1 / |N_r(dst)|, or w / |N_r(dst)| with edge_weight (E,)
    weight: torch.Tensor       # raw fuzzy degree (E,)
    num_relations: int
    sem: torch.Tensor | None   # (n, D) frozen semantic embeddings
    key_index: torch.Tensor    # edge -> its (relation, receiver) group (E,)
    key_dst: torch.Tensor      # receiver of each group, groups sorted by relation (K,)
    rel_ptr: tuple[int, ...]   # groups of relation r are key_dst[rel_ptr[r]:rel_ptr[r+1]]

    def to(self, device) -> "GraphTensors":
        """Move every tensor field to a device.

        Args:
            device: Target torch device.

        Returns:
            A new ``GraphTensors`` on ``device``.
        """
        return GraphTensors(*(v.to(device) if torch.is_tensor(v) else v for v in self.__dict__.values()))


def structural_features(kg: FuzzyKG, weights: np.ndarray | None = None) -> np.ndarray:
    """Inductive input features h^(0) from relation-wise weight statistics.

    Per relation: log(1 + Σ out-weights), max out-weight, and the same for
    incoming edges; columns are standardised.

    Args:
        kg: The knowledge graph.
        weights: Optional degrees overriding ``kg.weights``.

    Returns:
        Float32 array (n, 4|R|).
    """
    w = kg.weights if weights is None else weights
    n, R = kg.num_entities, kg.num_relations
    h, r, t = kg.triples.T
    feats = np.zeros((n, 4 * R), dtype=np.float32)
    np.add.at(feats, (h, r), w)
    np.maximum.at(feats, (h, R + r), w)
    np.add.at(feats, (t, 2 * R + r), w)
    np.maximum.at(feats, (t, 3 * R + r), w)
    feats[:, :R] = np.log1p(feats[:, :R])
    feats[:, 2 * R:3 * R] = np.log1p(feats[:, 2 * R:3 * R])
    std = feats.std(0)
    return (feats - feats.mean(0)) / np.where(std > 0, std, 1.0)


def build_graph(kg: FuzzyKG, sem: np.ndarray | None, inverse: bool = False, edge_weight: bool = False,
                weights: np.ndarray | None = None) -> GraphTensors:
    """Build the message-passing tensors.

    Paper default: messages flow from successors to an entity along the KG
    relations only, normalised by 1/|N_r(i)|.

    Args:
        kg: The knowledge graph.
        sem: Semantic embeddings (n, D), or None.
        inverse: Also pass messages along reversed edges (extra relation types).
        edge_weight: Scale messages by the fuzzy degree.
        weights: Optional degrees overriding ``kg.weights``.

    Returns:
        A ``GraphTensors``.
    """
    w = (kg.weights if weights is None else weights).astype(np.float32)
    h, r, t = kg.triples.T
    R = kg.num_relations
    src, dst, rel, ww = [t], [h], [r], [w]        # h aggregates from its successors t
    if inverse:
        src.append(h); dst.append(t); rel.append(r + R); ww.append(w)
    src, dst, rel, ww = (np.concatenate(v) for v in (src, dst, rel, ww))
    num_rel = 2 * R if inverse else R
    n = kg.num_entities
    key = rel.astype(np.int64) * n + dst                   # sorted by relation, then receiver
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    norm = (ww if edge_weight else np.ones_like(ww)) / cnt[inv]
    rel_ptr = tuple(int(v) for v in np.searchsorted(uniq // n, np.arange(num_rel + 1)))
    return GraphTensors(
        x=torch.as_tensor(structural_features(kg, weights)),
        src=torch.as_tensor(src), dst=torch.as_tensor(dst), rel=torch.as_tensor(rel),
        norm=torch.as_tensor(norm, dtype=torch.float32), weight=torch.as_tensor(ww),
        num_relations=num_rel, sem=None if sem is None else torch.as_tensor(sem, dtype=torch.float32),
        key_index=torch.as_tensor(inv.reshape(-1)), key_dst=torch.as_tensor(uniq % n), rel_ptr=rel_ptr)


# --------------------------------------------------------------------------- model

class RGCNLayer(nn.Module):
    """R-GCN layer with a full W_r per relation (Section IV-D).

    h_i' = W_0 h_i + Σ_r Σ_{j∈N_r(i)} norm_ij W_r h_j.
    """

    def __init__(self, d_in: int, d_out: int, num_relations: int):
        """Create the layer.

        Args:
            d_in: Input dimension.
            d_out: Output dimension.
            num_relations: Number of relation types.
        """
        super().__init__()
        self.self_loop = nn.Linear(d_in, d_out, bias=False)
        self.weight = nn.Parameter(torch.empty(num_relations, d_in, d_out))
        for w in self.weight:
            nn.init.xavier_uniform_(w)

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        # Normalised sum per (relation, receiver) group first, then one W_r product per relation.
        """Apply the layer.

        Neighbour messages are first summed per (relation, receiver) group, then
        multiplied by W_r once per relation.

        Args:
            h: Node states (n, d_in).
            g: Graph tensors.

        Returns:
            New node states (n, d_out), before the activation.
        """
        agg = h.new_zeros(len(g.key_dst), h.shape[1]).index_add_(0, g.key_index, g.norm[:, None] * h[g.src])
        sizes = [hi - lo for lo, hi in zip(g.rel_ptr[:-1], g.rel_ptr[1:])]
        msg = torch.cat([a @ w for a, w in zip(torch.split(agg, sizes), self.weight.unbind(0)) if len(a)])
        return self.self_loop(h).index_add_(0, g.key_dst, msg)


class WeightedRGCNLayer(nn.Module):
    """R-GCN layer with basis decomposition W_r = Σ_b a_rb V_b.

    h_i' = W_0 h_i + Σ_r Σ_{j∈N_r(i)} norm_ij W_r h_j.
    """

    def __init__(self, d_in: int, d_out: int, num_relations: int, num_bases: int):
        """Create the layer.

        Args:
            d_in: Input dimension.
            d_out: Output dimension.
            num_relations: Number of relation types.
            num_bases: Number of basis matrices V_b.
        """
        super().__init__()
        self.self_loop = nn.Linear(d_in, d_out)
        self.bases = nn.Parameter(torch.empty(num_bases, d_in, d_out))
        self.coeff = nn.Parameter(torch.empty(num_relations, num_bases))
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeff)

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        """Apply the layer.

        Args:
            h: Node states (n, d_in).
            g: Graph tensors.

        Returns:
            New node states (n, d_out), before the activation.
        """
        hs = h[g.src]                                         # gathered once, shared by all bases
        out = self.self_loop(h)
        for b in range(self.bases.shape[0]):
            coef = (g.norm * self.coeff[g.rel, b]).unsqueeze(1)
            agg = torch.zeros_like(h).index_add_(0, g.dst, coef * hs)
            out = out + agg @ self.bases[b]
        return out


class GraphEncoder(nn.Module):
    """Structural encoder g: input projection followed by message-passing layers."""
    def __init__(self, kind: str, feat_dim: int, num_relations: int, d: int = 128, layers: int = 2,
                 num_bases: int = 0, dropout: float = 0.2, final_relu: bool = False):
        """Create the encoder.

        Args:
            kind: ``"rgcn"``, ``"gcn"``, ``"sage"`` or ``"mlp"``.
            feat_dim: Input feature dimension.
            num_relations: Number of edge relation types (R-GCN only).
            d: Hidden and output dimension.
            layers: Number of message-passing layers.
            num_bases: 0 for a full W_r per relation, K > 0 for K bases (R-GCN only).
            dropout: Dropout rate before every layer.
            final_relu: Apply ReLU after the last layer, as in the paper.  Off by
                default because non-negative embeddings bound B̂ >= 0.5.

        Raises:
            ValueError: If ``kind`` is unknown.
        """
        super().__init__()
        self.kind = kind
        self.inp = nn.Linear(feat_dim, d)
        self.dropout = dropout
        self.final_relu = final_relu
        if kind == "rgcn" and num_bases > 0:
            self.layers = nn.ModuleList(WeightedRGCNLayer(d, d, num_relations, num_bases) for _ in range(layers))
        elif kind == "rgcn":
            self.layers = nn.ModuleList(RGCNLayer(d, d, num_relations) for _ in range(layers))
        elif kind == "gcn":
            from torch_geometric.nn import GCNConv
            self.layers = nn.ModuleList(GCNConv(d, d) for _ in range(layers))
        elif kind == "sage":
            from torch_geometric.nn import SAGEConv
            self.layers = nn.ModuleList(SAGEConv(d, d) for _ in range(layers))
        elif kind == "mlp":
            self.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        else:
            raise ValueError(kind)

    def forward(self, g: GraphTensors) -> torch.Tensor:
        """Compute structural embeddings of all entities.

        Args:
            g: Graph tensors.

        Returns:
            Embeddings (n, d).
        """
        h = F.relu(self.inp(g.x))
        edge_index = torch.stack([g.src, g.dst])
        for k, layer in enumerate(self.layers):
            h = F.dropout(h, self.dropout, self.training)
            if self.kind == "rgcn":
                h = layer(h, g)
            elif self.kind == "gcn":
                h = layer(h, edge_index, g.weight)
            elif self.kind == "sage":
                h = layer(h, edge_index)
            else:
                h = layer(h)
            # Paper: σ = ReLU on every layer; off on the last one by default (B̂ >= 0.5 otherwise).
            if k < len(self.layers) - 1 or self.final_relu:
                h = F.relu(h)
        return h


class BisimNet(nn.Module):
    """Bisimulation approximator B̂(e_i, e_j) = (cos(z_i, z_j) + 1) / 2.

    z = α · g(e) + (1 - α) · ϕ_proj(e) (Section IV-E).
    """
    def __init__(self, encoder: str, feat_dim: int, num_relations: int, sem_dim: int | None,
                 d: int = 128, layers: int = 2, num_bases: int = 0, dropout: float = 0.2,
                 alpha: float | str = 0.7, normalize_fusion: bool = False, final_relu: bool = False):
        """Create the model.

        Args:
            encoder: Graph encoder kind, see ``GraphEncoder``.
            feat_dim: Structural feature dimension.
            num_relations: Number of edge relation types.
            sem_dim: Semantic embedding dimension, or None without a semantic channel.
            d: Embedding dimension.
            layers: Number of message-passing layers.
            num_bases: Basis count for the R-GCN, 0 for full matrices.
            dropout: Dropout rate.
            alpha: Fusion weight in [0, 1], or ``"learn"`` for a learned sigmoid weight.
                α = 1 drops the semantic channel, α = 0 drops the graph encoder.
            normalize_fusion: L2-normalise both channels before fusion.
            final_relu: ReLU after the last GNN layer, see ``GraphEncoder``.
        """
        super().__init__()
        self.alpha_mode = alpha
        self.use_graph = alpha == "learn" or float(alpha) > 0.0
        self.use_sem = sem_dim is not None and (alpha == "learn" or float(alpha) < 1.0)
        self.graph = GraphEncoder(encoder, feat_dim, num_relations, d, layers, num_bases, dropout,
                                  final_relu) \
            if self.use_graph else None
        self.proj = nn.Linear(sem_dim, d) if self.use_sem else None
        if alpha == "learn":
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.normalize_fusion = normalize_fusion

    @property
    def alpha(self) -> float | torch.Tensor:
        """Current fusion weight α (a tensor when learned)."""
        if self.alpha_mode == "learn":
            return torch.sigmoid(self.alpha_logit)
        return float(self.alpha_mode) if self.use_sem else 1.0

    def embed(self, g: GraphTensors) -> torch.Tensor:
        """Fused embeddings z of all entities.

        Args:
            g: Graph tensors (``g.sem`` is used by the semantic channel).

        Returns:
            Embeddings (n, d).
        """
        norm = (lambda v: F.normalize(v, dim=-1)) if self.normalize_fusion else (lambda v: v)
        parts = []
        if self.graph is not None:
            parts.append((self.alpha if self.use_sem else 1.0, norm(self.graph(g))))
        if self.proj is not None:
            parts.append(((1 - self.alpha) if self.graph is not None else 1.0, norm(self.proj(g.sem))))
        return sum(c * v for c, v in parts)

    @staticmethod
    def similarity(zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        """Predicted bisimulation degree σ_sim(cos(z_i, z_j)) = (cos + 1) / 2.

        Args:
            zi: Embeddings of the first entities (..., d).
            zj: Embeddings of the second entities (..., d).

        Returns:
            Degrees in [0, 1].
        """
        return (F.cosine_similarity(zi, zj, dim=-1) + 1.0) / 2.0
