"""Fuzzy knowledge graph loading and confidence assignment.

A fuzzy KG is G = (E, R, T, w) with w : T -> [0, 1] (Section III-B).  Benchmarks
without confidences get them from a *confidence scheme* (Section V-A uses
Beta(2, 2)); the extra schemes exist so the effect of that assumption can be
measured (reviewer comment on artificial uncertainty).
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import DATA_ROOT

KGBERT = "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data"
CONVE = "https://raw.githubusercontent.com/TimDettmers/ConvE/master"


def _kgbert(name: str) -> dict:
    files = ["train.tsv", "dev.tsv", "test.tsv", "entity2text.txt", "relation2text.txt"]
    return {"files": {f: f"{KGBERT}/{name}/{f}" for f in files},
            "splits": ["train.tsv", "dev.tsv", "test.tsv"]}


# The three benchmarks of the paper (Section V-A).
DATASETS: dict[str, dict] = {
    "fb15k-237": _kgbert("FB15k-237"),
    "wn18rr": _kgbert("WN18RR"),
    "yago3-10": {"tar": f"{CONVE}/YAGO3-10.tar.gz",
                 "splits": ["train.txt", "valid.txt", "test.txt"]},
}


@dataclass
class FuzzyKG:
    name: str
    entities: list[str]
    relations: list[str]
    triples: np.ndarray                     # (m, 3) int64 rows (h, r, t)
    weights: np.ndarray                     # (m,) float32 in [0, 1]
    given_weights: np.ndarray | None = None  # confidences shipped with the data
    entity_text: list[str] | None = None
    relation_text: list[str] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    @property
    def num_triples(self) -> int:
        return len(self.triples)

    def degree(self) -> np.ndarray:
        """Total (in + out) degree of every entity."""
        n = self.num_entities
        return (np.bincount(self.triples[:, 0], minlength=n)
                + np.bincount(self.triples[:, 2], minlength=n))

    def summary(self) -> str:
        return (f"{self.name}: {self.num_entities} entities, {self.num_relations} relations, "
                f"{self.num_triples} triples, w mean={self.weights.mean():.3f} "
                f"std={self.weights.std():.3f}")


# --------------------------------------------------------------------------- download

def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def dataset_folder(name: str, root: Path = DATA_ROOT) -> Path:
    """data/<name>, matching an existing folder case-insensitively
    (``fb15k-237`` finds ``data/FB15k-237``)."""
    if root.exists():
        for p in root.iterdir():
            if p.is_dir() and p.name.lower() == name.lower():
                return p
    return root / name


def triple_files(folder: Path) -> list[Path]:
    """Triple files in a dataset folder: *.txt / *.tsv except the *2text files."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix in {".tsv", ".txt"} and "2text" not in p.name)


def download_dataset(name: str, root: Path = DATA_ROOT) -> Path:
    """Use triple files already in data/<name>/; download only what is missing.

    If the folder already holds triples (e.g. just train.txt), no split is
    downloaded; only the entity / relation name files, which the semantic
    channel needs, are fetched when absent.
    """
    folder = dataset_folder(name, root)
    spec = DATASETS.get(name.lower())
    if spec is None:
        return folder
    if triple_files(folder):
        for fname, url in spec.get("files", {}).items():
            if "2text" in fname and not (folder / fname).exists():
                try:
                    _fetch(url, folder / fname)
                except OSError as exc:
                    print(f"  could not fetch {fname} ({exc}); entity names fall back to ids")
        return folder
    if "tar" in spec:
        archive = folder / "archive.tar.gz"
        _fetch(spec["tar"], archive)
        with tarfile.open(archive) as tar:
            # Only pull the expected split files out of the downloaded archive.
            for member in tar.getmembers():
                base = Path(member.name).name
                if member.isfile() and base in spec["splits"]:
                    member.name = base
                    tar.extract(member, folder, filter="data")
        archive.unlink()
    else:
        for fname, url in spec["files"].items():
            if not (folder / fname).exists():
                _fetch(url, folder / fname)
    return folder


# --------------------------------------------------------------------------- loading

def _read_rows(path: Path) -> list[list[str]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append(parts)
    return rows


def _read_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def load_kg(name: str, root: Path = DATA_ROOT, confidence: str = "auto",
            seed: int = 0) -> FuzzyKG:
    """Load the union of all splits as one fuzzy KG and assign confidences.

    Link-prediction splits are irrelevant here: bisimulation is defined on the
    whole graph, so train/valid/test triples are merged.  Files may have three
    columns (h, r, t) or four (h, r, t, w); with four, ``confidence='given'``
    uses the shipped scores.
    """
    folder = download_dataset(name, root)
    files = triple_files(folder)
    if not files:
        raise FileNotFoundError(f"no triple files (*.txt / *.tsv) in {folder}")
    rows = [r for f in files for r in _read_rows(f)]

    # entities.dict ("<id>\t<name>"), when present, fixes the entity vocabulary
    # so entities that only occur in absent valid/test splits still count.
    # Relations always come from the triples.
    ent_index = {e: k for k, e in enumerate(_read_mapping(folder / "entities.dict").values())}
    rel_index: dict[str, int] = {}
    triples = np.empty((len(rows), 3), dtype=np.int64)
    given = np.full(len(rows), np.nan, dtype=np.float32)
    for k, row in enumerate(rows):
        h, r, t = row[:3]
        triples[k] = (ent_index.setdefault(h, len(ent_index)),
                      rel_index.setdefault(r, len(rel_index)),
                      ent_index.setdefault(t, len(ent_index)))
        if len(row) >= 4:
            given[k] = float(row[3])

    # Deduplicate (h, r, t), keeping the highest shipped confidence.
    order = np.lexsort((-np.nan_to_num(given, nan=-1.0), triples[:, 2], triples[:, 1], triples[:, 0]))
    triples, given = triples[order], given[order]
    keep = np.ones(len(triples), dtype=bool)
    keep[1:] = np.any(triples[1:] != triples[:-1], axis=1)
    triples, given = triples[keep], given[keep]

    entities = list(ent_index)
    relations = list(rel_index)
    ent_text = _read_mapping(folder / "entity2text.txt")
    rel_text = _read_mapping(folder / "relation2text.txt")
    kg = FuzzyKG(
        name=name,
        entities=entities,
        relations=relations,
        triples=triples,
        weights=np.ones(len(triples), dtype=np.float32),
        given_weights=None if np.isnan(given).all() else np.nan_to_num(given, nan=1.0).clip(0, 1),
        entity_text=[ent_text.get(e, e) for e in entities] if ent_text else None,
        relation_text=[rel_text.get(r, r) for r in relations] if rel_text else None,
    )
    kg.weights = assign_confidence(kg, confidence, seed)
    kg.meta["confidence"] = confidence
    return kg


# --------------------------------------------------------------------------- confidence

CONFIDENCE_SCHEMES = """
  auto              'given' when the data ships confidences, else 'beta:2,2'
  given             confidences shipped in a 4th column of the data files
  beta:a,b          w ~ Beta(a, b), i.i.d. per triple (paper: Beta(2,2))
  uniform           w ~ U(0, 1)
  crisp             w = 1 (fuzzy bisimulation collapses to crisp bisimulation)
  pathsupport       deterministic, data-derived: noisy-OR over corroborating
                    2-hop paths between h and t
  relfreq           deterministic, data-derived: w = 1 / (number of tails the head
                    has for that relation)^0.5, i.e. functional facts are surer
"""


def assign_confidence(kg: FuzzyKG, scheme: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = kg.num_triples
    if scheme == "auto":
        scheme = "given" if kg.given_weights is not None else "beta:2,2"
    if scheme == "given":
        if kg.given_weights is None:
            raise ValueError(f"{kg.name} ships no confidence scores; pick another scheme")
        w = kg.given_weights
    elif scheme.startswith("beta"):
        a, b = (float(x) for x in scheme.split(":")[1].split(","))
        w = rng.beta(a, b, size=m)
    elif scheme == "uniform":
        w = rng.uniform(0.0, 1.0, size=m)
    elif scheme == "crisp":
        w = np.ones(m)
    elif scheme == "pathsupport":
        w = _path_support(kg)
    elif scheme == "relfreq":
        h, r = kg.triples[:, 0], kg.triples[:, 1]
        key = h * kg.num_relations + r
        _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        w = 1.0 / np.sqrt(counts[inv])
    else:
        raise ValueError(f"unknown confidence scheme {scheme!r}; choose from:{CONFIDENCE_SCHEMES}")
    # Gödel semantics only uses order and equality of degrees; rounding keeps the
    # fixpoint's value set small without changing the semantics noticeably.
    return np.round(np.asarray(w, dtype=np.float64), 3).clip(0.0, 1.0).astype(np.float32)


def _path_support(kg: FuzzyKG, p0: float = 0.3, p: float = 0.15) -> np.ndarray:
    """w = 1 - (1 - p0)(1 - p)^k where k = #common neighbours of h and t."""
    n = kg.num_entities
    h, t = kg.triples[:, 0], kg.triples[:, 2]
    adj = sp.coo_matrix((np.ones(len(h)), (h, t)), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    k = np.zeros(len(h))
    for start in range(0, len(h), 200_000):
        sl = slice(start, start + 200_000)
        k[sl] = np.asarray(adj[h[sl]].multiply(adj[t[sl]]).sum(axis=1)).ravel()
    return 1.0 - (1.0 - p0) * (1.0 - p) ** k


def perturb_weights(w: np.ndarray, delta: float, seed: int = 0) -> np.ndarray:
    """Shift every degree by at most ``delta`` (used for the Theorem 2 check)."""
    rng = np.random.default_rng(seed)
    return (w + rng.uniform(-delta, delta, size=w.shape)).clip(0.0, 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download / inspect fuzzy KG datasets.")
    ap.add_argument("datasets", nargs="+", choices=list(DATASETS))
    ap.add_argument("--confidence", default="auto", help=CONFIDENCE_SCHEMES)
    args = ap.parse_args()
    for name in args.datasets:
        kg = load_kg(name, confidence=args.confidence)
        print(kg.summary())


if __name__ == "__main__":
    main()
