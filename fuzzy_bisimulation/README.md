# Learning fuzzy bisimulation on fuzzy knowledge graphs

Code for *Learning Behavioral Equivalence in Fuzzy Knowledge Graphs via
Neuro-Symbolic Methods for Smart Grid Fault Diagnosis*, revised for the review
comments.

## Pipeline

| Stage | Module | Paper |
|---|---|---|
| Fuzzy KG loading, confidence schemes | `fuzzy_kg.py` | III-B, V-A |
| Exact fuzzy bisimulation (Eq. 1, Algorithm 1) | `exact_bisimulation.py` | III-D, IV-C |
| Degree-aware sampling, local exact references | `local_sampling.py` | IV-C, V-A |
| Supervision + evaluation references, offline timings | `generate_labels.py` | IV-C |
| Entity descriptions and semantic embeddings | `semantic.py` | III-E, IV-E |
| Weighted R-GCN, α-fusion (model) | `model.py` | IV-D, IV-E |
| Training loop, Algorithm 2 | `train.py` | IV-D, IV-E |
| Metrics, cost split, baselines, statistics | `evaluate.py` | V-C, VI |
| Experiment suites | `experiments.py` | VI + review |

```bash
pip install -r requirements.txt
python -m fuzzy_bisimulation.exact_bisimulation --selftest          # vectorised F == literal Eq. (1)

python -m fuzzy_bisimulation.generate_labels --dataset fb15k-237    # stage 1
python -m fuzzy_bisimulation.train --labels data/FB15k-237/labels/<tag>.pt --semantic openai:text-embedding-3-large:768
python -m fuzzy_bisimulation.evaluate --checkpoint checkpoints/<run>.pt
python -m fuzzy_bisimulation.experiments main --dataset fb15k-237   # 5 seeds, all baselines
```

Datasets: `fb15k-237`, `wn18rr`, `yago3-10` (folder names are matched without
regard to case, e.g. `data/FB15k-237/`). Every `*.txt` / `*.tsv` file in the
folder except `*2text*` is read as triples (`h r t [w]`); `entities.dict` fixes
the entity vocabulary, so counts match Table II (14,541 / 40,943 / 123,182
entities; 272,115 / 86,835 / 1,079,040 triples from `train.txt`). Missing
`entity2text.txt` / `relation2text.txt` (entity names for the semantic
channel) are fetched from the KG-BERT copies for FB15k-237 and WN18RR; YAGO3-10
ids are already readable names. If a folder has no triples, the splits are
downloaded.

## Implementation notes

* **Operator.** One application of F costs O(|T|·n) time and O(n²) memory: it
  loops over transitions instead of forming dense max-min products. The
  self-test checks it against a literal transcription of Eq. (1) for every
  semantics.
* **State labels.** `--label-mode inout` (the default) uses one proposition per
  relation and direction, L(s, p_r) = the largest weight on an r-edge. Without
  incoming propositions, every entity that has no outgoing edges would be
  bisimilar to every other such entity.
* **Message direction.** The R-GCN aggregates from successors, because
  bisimulation is decided by the moves s → s′. Inverse edges are added as
  separate relation types. Messages are scaled by w/|N_r(i)|.
* **No ReLU on the output layer.** With non-negative embeddings, cos ≥ 0, so B̂
  could never fall below 0.5.
* **Normalised fusion.** ĝ and ŝ are L2-normalised before
  z = αĝ + (1−α)ŝ. Without this, α is mixed up with the norms of the two
  channels.
* **Inductive inputs.** Node features are per-relation in/out weight
  statistics. Held-out entities therefore get embeddings without any learned
  per-entity parameters. KGE baselines (TransE, …) are not included, because
  embedding tables cannot score entities that never got supervision.

## Findings from running the code

1. **Gödel B\* is almost binary on real graphs.** Under Eq. (1),
   δ ⇒ 0 = 0. One unmatched transition at any depth sets the degree to 0. For
   FB15k-237 (sample of 1,454 entities, Beta(2,2)), the mean B\* is 0.004 and
   0.4% of pairs reach 0.7. A regression target like
   this is degenerate, and F1 at τ = 0.7 is undefined on uniform pairs. Two
   things follow:
   * `--discount λ` and `--semantics lukasiewicz|product` are options that
     keep F monotone (`experiments semantics` reports the value distributions).
   * Metrics are reported on uniform pairs **and** on value-stratified pairs.
2. **Assumption 1 fails for induced subgraphs.** Dropping the edges that leave
   E_s turns many entities into sinks, which inflates B\*_s. In early tests,
   degree-aware sampling roughly halved E|B\*_s − B\*| compared with random
   sampling, but did not remove the gap; `--train-reference local` does.
   `experiments reference` measures the gap per dataset.
3. **Local references are exact.** On the induced k-ball, B^(k) equals the
   full-graph B^(k) with zero error (checked against a full-graph run). The
   catch on dense graphs: on FB15k-237 the 2-step ball around only 100 test
   entities already covers 9,204 of 14,541 nodes, so the saving there is
   small. Report this honestly rather than claiming large-graph scalability.

## Response to the reviewers: where each point is addressed

| Comment | Change / evidence |
|---|---|
| Scalability; full-graph B\* used as evaluation reference | `--eval-reference local` (default) computes the *exact* full-graph B^(k) from a k-ball around the test entities, with no full-graph fixpoint. A complete forward closure gives exact B\*. `experiments reference` measures local versus full accuracy and time. `evaluate.py` separates offline cost (sampling, labels, embeddings, training) from per-query cost and reports the break-even number of queries. |
| Beta(2,2) confidences are artificial | `--confidence` accepts `given` (a 4th column in the data files), `beta:a,b`, `uniform`, `crisp`, and the data-derived `pathsupport` and `relfreq`. `experiments confidence` reports sensitivity to the choice. |
| LLM choice and fixed α = 0.7 | `experiments alpha` sweeps α from 0 to 1 and adds a learned α. `experiments encoders` compares OpenAI, sentence-transformers and TF-IDF encoders. Fusion is normalised so α has a well-defined meaning. |
| "Constant-time inference" is misleading | `experiments latency` shows query cost is linear in d (≈ 43 ns at d = 16 → 5.5 µs at d = 2048, CPU). In the text, say "O(d) per query given precomputed embeddings, independent of \|E\| for fixed d". |
| Novelty | Methodological: exact local references, a supervision protocol that removes the induced-subgraph bias, and a measured cost split. None of these is a new operator or architecture. |
| Requested citations | The editor asked authors to disregard them. |
| Smart-grid case study details (R1, R2) | No grid data are available, so the case study is removed; the paper and the response letter must say so. |
| Theoretical assumptions not verified (R1, R2) | `experiments assumptions` measures eps_s (Assumption 1), eps_nn (Assumption 2), the Lipschitz constant of the projection (Assumption 3: spectral norm and observed maximum), Theorem 1's bound against train and held-out error, and `stability` gives Theorem 2's constant. Assumption 3 as printed uses \|e_i - e_j\|, which is undefined for entities; the check uses phi. |
| Critical limitations; formatting/equations (R1) | Text only. Material for limitations: Gödel degeneracy, induced-subgraph bias, local balls approaching the whole graph on dense KGs, and the break-even query counts. Eq. (1) has a typo: R(s',t) in the backward term should be R(s',t'). |

## Experiment suites

```bash
python -m fuzzy_bisimulation.experiments {main,ablation,sampling,latency,reference,train-ref,confidence,
                                          alpha,encoders,semantics,stability,assumptions} \
       --dataset <name> [--semantics lukasiewicz --discount 0.7] [--train-reference local] \
       [--semantic openai:text-embedding-3-large:768] [--seeds 0 1 2 3 4]
```

Outputs go to `results/<suite>_<dataset>.{json,md}`.

Runtime: training does one full-graph forward pass per step, which is slow on
CPU for these graphs. Use a GPU (`--device cuda`) for
FB15k-237 and YAGO3-10, and tune `--steps-per-epoch`.
