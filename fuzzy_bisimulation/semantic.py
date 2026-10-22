"""Semantic entity embeddings φ(e) from textual descriptions (Section III-E, IV-E).

Each entity gets a description built from its name and its relations, encoded by
one of several interchangeable encoders, so the choice of
text-embedding-3-large can be compared with alternatives (reviewer comment):

  openai:<model>[:dim]  OpenAI embeddings API (needs OPENAI_API_KEY); paper:
                        openai:text-embedding-3-large:768
  st:<model>            sentence-transformers, e.g. st:all-MiniLM-L6-v2,
                        st:all-mpnet-base-v2, st:BAAI/bge-small-en-v1.5
  hf:<model>            HuggingFace transformers with mean pooling
  tfidf[:dim]           hashed word/bigram TF-IDF + Gaussian projection; no
                        model, no download, and a lexical baseline
  none                  no semantic channel

Embeddings are cached in data/<dataset>/semantic/ together with their
generation time (an offline cost reported by evaluate.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import DATA_ROOT
from .fuzzy_kg import FuzzyKG, dataset_folder, load_kg

DESCRIPTION_VERSION = "v1"


def _pretty(name: str) -> str:
    """Readable form of a relation id such as ``/people/person/nationality``.

    Args:
        name: Relation identifier.

    Returns:
        The last path segment with separators replaced by spaces.
    """
    name = name.strip("/").split("/")[-1] if name.startswith("/") else name
    return re.sub(r"[_./]+", " ", name).strip()


def build_descriptions(kg: FuzzyKG, max_facts: int = 8) -> list[str]:
    """Textual description of every entity for the semantic encoder.

    Format: ``'<name>. <relation> <tail>; ...; <head> <relation> it; ...'``.
    Facts are ordered by confidence so the most certain ones are kept.

    Args:
        kg: The knowledge graph.
        max_facts: Maximum number of facts per entity (half outgoing, half incoming).

    Returns:
        One description per entity, in entity-index order.
    """
    names = kg.entity_text or kg.entities
    rels = kg.relation_text or [_pretty(r) for r in kg.relations]
    order = np.argsort(-kg.weights, kind="stable")
    out_facts: list[list[str]] = [[] for _ in range(kg.num_entities)]
    in_facts: list[list[str]] = [[] for _ in range(kg.num_entities)]
    half = max(1, max_facts // 2)
    for k in order:
        h, r, t = kg.triples[k]
        if len(out_facts[h]) < half:
            out_facts[h].append(f"{rels[r]} {names[t].split(',')[0]}")
        if len(in_facts[t]) < half:
            in_facts[t].append(f"{names[h].split(',')[0]} {rels[r]} it")
    return [f"{names[e]}. " + "; ".join(out_facts[e] + in_facts[e]) for e in range(kg.num_entities)]


# --------------------------------------------------------------------------- encoders

def _tfidf(texts: list[str], dim: int = 768, buckets: int = 1 << 15, seed: int = 0) -> np.ndarray:
    """Hashed word/bigram TF-IDF followed by a Gaussian random projection.

    Args:
        texts: Input texts.
        dim: Output dimension.
        buckets: Number of hash buckets.
        seed: Seed of the projection matrix.

    Returns:
        L2-normalised float32 array (len(texts), dim).
    """
    rows, cols = [], []
    for i, text in enumerate(texts):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for g in grams:
            rows.append(i)
            cols.append(int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % buckets)
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(texts), buckets))
    X.sum_duplicates()
    X.data = 1.0 + np.log(X.data)                                        # sublinear tf
    df = np.bincount(X.indices, minlength=buckets)
    X = X @ sp.diags(np.log((1 + len(texts)) / (1 + df)).astype(np.float32) + 1.0)
    G = np.random.default_rng(seed).standard_normal((buckets, dim)).astype(np.float32) / np.sqrt(dim)
    E = np.asarray(X @ G)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def _sentence_transformers(texts: list[str], model: str, device: str | None) -> np.ndarray:
    """Encode texts with a sentence-transformers model.

    Args:
        texts: Input texts.
        model: Model name, e.g. ``all-mpnet-base-v2``.
        device: Torch device, or None for the library default.

    Returns:
        L2-normalised embeddings (len(texts), D).
    """
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model, device=device)
    return enc.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)


def _huggingface(texts: list[str], model: str, device: str | None) -> np.ndarray:
    """Encode texts with a HuggingFace model using attention-masked mean pooling.

    Args:
        texts: Input texts.
        model: Model name on the HuggingFace hub.
        device: Torch device, or None to pick CUDA when available.

    Returns:
        L2-normalised embeddings (len(texts), D).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModel.from_pretrained(model).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128,
                        return_tensors="pt").to(device)
            h = net(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    E = np.concatenate(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def _openai(texts: list[str], model: str, dim: int | None) -> np.ndarray:
    """Encode texts with the OpenAI embeddings API.

    Args:
        texts: Input texts (truncated to 8,000 characters each).
        model: Embedding model, e.g. ``text-embedding-3-large``.
        dim: Requested output dimension, or None for the model default.

    Returns:
        Float32 embeddings (len(texts), D).

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is not set.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY to use an openai: encoder")
    out = []
    for i in range(0, len(texts), 256):
        body = {"model": model, "input": [t[:8000] for t in texts[i:i + 256]]}
        if dim:
            body["dimensions"] = dim
        req = urllib.request.Request("https://api.openai.com/v1/embeddings",
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)["data"]
        out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
    return np.asarray(out, dtype=np.float32)


def encode(texts: list[str], spec: str, device: str | None = None) -> np.ndarray:
    """Encode texts with the encoder named by ``spec``.

    Args:
        texts: Input texts.
        spec: ``tfidf[:dim]``, ``st:<model>``, ``hf:<model>`` or
            ``openai:<model>[:dim]``.
        device: Torch device for local models.

    Returns:
        Embeddings (len(texts), D).

    Raises:
        ValueError: If the encoder kind is unknown.
    """
    kind, _, rest = spec.partition(":")
    if kind == "tfidf":
        return _tfidf(texts, dim=int(rest) if rest else 768)
    if kind == "st":
        return _sentence_transformers(texts, rest, device)
    if kind == "hf":
        return _huggingface(texts, rest, device)
    if kind == "openai":
        model, _, dim = rest.partition(":")
        return _openai(texts, model, int(dim) if dim else None)
    raise ValueError(f"unknown encoder {spec!r}")


def cache_path(dataset: str, spec: str, root: Path = DATA_ROOT) -> Path:
    """Cache file of the embeddings for a dataset and encoder.

    Args:
        dataset: Dataset name.
        spec: Encoder spec.
        root: Data root folder.

    Returns:
        ``data/<dataset>/semantic/<spec>_<version>.npy``.
    """
    safe = re.sub(r"[^A-Za-z0-9.-]+", "_", spec)
    return dataset_folder(dataset, root) / "semantic" / f"{safe}_{DESCRIPTION_VERSION}.npy"


def load_or_encode(kg: FuzzyKG, spec: str, root: Path = DATA_ROOT,
                   device: str | None = None) -> tuple[np.ndarray | None, float]:
    """Load cached semantic embeddings, or compute and cache them.

    Args:
        kg: The knowledge graph.
        spec: Encoder spec, or ``"none"``.
        root: Data root folder.
        device: Torch device for local models.

    Returns:
        ``(embeddings, seconds)``: float32 array (n, D) and the generation time
        (read from the cache metadata when cached); ``(None, 0.0)`` for ``none``.
    """
    if spec == "none":
        return None, 0.0
    path = cache_path(kg.name, spec, root)
    meta_path = path.with_suffix(".json")
    if path.exists():
        E = np.load(path)
        if len(E) == kg.num_entities:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            return E, float(meta.get("seconds", 0.0))
    t0 = time.perf_counter()
    E = encode(build_descriptions(kg), spec, device).astype(np.float32)
    seconds = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    meta_path.write_text(json.dumps({"spec": spec, "seconds": seconds, "shape": list(E.shape)}))
    return E, seconds


def main() -> None:
    """Command-line entry point: print sample descriptions and build the cache."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", default="tfidf")
    ap.add_argument("--show", type=int, default=3, help="print a few descriptions")
    args = ap.parse_args()
    kg = load_kg(args.dataset)
    for d in build_descriptions(kg)[:args.show]:
        print(" ", d[:200].encode("ascii", "replace").decode())
    E, sec = load_or_encode(kg, args.encoder)
    print(f"{args.encoder}: shape={E.shape} time={sec:.1f}s -> {cache_path(kg.name, args.encoder)}")


if __name__ == "__main__":
    main()
