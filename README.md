# OEP-CVR

### Semantic Web-Driven Ontology-Constrained Compositional Video Retrieval

**Research implementation · PyTorch · RDF/SPARQL · Split-conformal inference**

> **OEP-CVR** combines calibrated visual class sets, three-valued ontology conditions, non-compensatory partial-order ranking, and cost-aware observation review for fine-grained sports-video retrieval. The implementation follows the method and experimental configuration described in *Semantic Web-Driven Ontology-Constrained Compositional Video Retrieval*.

---

## 🧭 Overview

A composed-video query contains a **reference video** and a **textual action modification**. The system must retrieve videos satisfying the requested modification, including fine-grained distinctions between stages, movement names, units, and twist counts. When reference recognition or text interpretation is uncertain, a single predicted action class can introduce incorrect downstream constraints.

OEP-CVR explicitly carries unresolved interpretations into two semantic target sets. The **certain-target lower bound** contains classes supported by all valid reference/interpretation combinations, while the **possible-target upper bound** contains classes not excluded by at least one combination. A candidate is assigned *support*, *conflict*, or *unknown* evidence; confirmed support must precede confirmed conflict, and unknown evidence remains eligible for retrieval. A learned policy then selects additional observations of the reference or a candidate using predicted ranking gain divided by review cost.

```text
  Reference frames ──┐
                     ├── Frozen CLIP ViT-B/16 ── Temporal Transformer ── Class sets
  Gallery frames ────┘                   │                    │
                                       └── Composed query ────┴── Initial similarity
  Modification text ── CLIP text encoder ────────────────┘          │
                         │                                      Top-M fixed pool
                 Frozen finite grammar                              │
                         │            RDF + OWL class relations     │
                         └───────────── SPARQL bindings ────────────┤
                                           │                        │
                                   Lower/upper bounds ── Evidence states
                                                                    │
                           Value network ── Review decisions ── Partial-order ranking
                                                                    │
                                                            Ranked retrieval
```

### Method-to-code correspondence

| Scientific component | Implementation | Method correspondence |
|:--|:--|:--|
| Experimental settings and seed control | `config.py` | Configuration in *Experiments* |
| Video/query manifests and multi-target labels | `data.py` | Data partitions and evaluation protocol |
| Frozen CLIP interface, temporal encoder, composition head | `model.py` | Temporal and semantic representation learning |
| Ontology attributes, subclass closure, grammar, and three-valued conditions | `semantics.py` | Eqs. (3)–(5) |
| Joint-budget calibration, partial order, and ranking losses | `reasoning.py` | Eqs. (1), (2), (6)–(8), (12), (13) |
| Reference/candidate review, policy inputs, and relational gain | `scheduling.py` | Eqs. (9), (10) |
| Three-stage optimization, inference CLI, evaluation, and checks | `run.py` | Eq. (11) and Algorithm 1 |

The repository consists of **seven Python source files and this README**. No additional package-specific runtime files are required.

---

## 🧠 Mathematical specification

**Joint-budget class calibration.** For the four- and twelve-slot paths, nonconformity uses the worst observed budget:

$$
a_j = \max_{b\in\{4,12\}}\left[1-p_b(y_j\mid v_j)\right],\qquad
k=\left\lceil(n+1)(1-\alpha)\right\rceil.
$$

The threshold is the $k$th ordered score when $k\le n$, and $+\infty$ otherwise. After a deep review, the prediction sets from the two budgets are intersected. Empty intersections revert to the complete class vocabulary with unknown evidence. The default target miscoverage is $\alpha=0.10$.

**Dual semantic target sets.** Given calibrated reference classes $S_r$ and valid programs $\Pi(m)$:

$$
A^-_q = \bigcap_{(z_r,\pi)\in S_r\times\Pi(m)} E^-_{\mathcal O}(z_r,\pi),\qquad
A^+_q = \bigcup_{(z_r,\pi)\in S_r\times\Pi(m)} E^+_{\mathcal O}(z_r,\pi).
$$

Each atomic condition evaluates to `true`, `false`, or `unknown`. Unknown attributes are not treated as negative assertions. Unparseable, contradictory, or out-of-vocabulary queries use a documented similarity-only fallback.

**Partial-order retrieval.** For candidate class set $S_i$:

- **Support:** $S_i\subseteq A^-_q$.
- **Conflict:** $S_i\cap A^+_q=\varnothing$.
- **Unknown:** all remaining or invalid evidence states.

The directed precedence graph contains a support-to-conflict edge for every such pair. Inference repeatedly selects the highest-scoring zero-indegree candidate, resolving ties through initial rank and candidate identifier. The selected pool remains fixed during review; gallery items outside the pool retain their initial order.

**Budgeted review.** The policy maps twelve normalized decision features through a `12 → 64 → 1` network. Each action is selected by positive predicted relational gain per positive review cost. A reference review updates the shared query, bounds, and **all** pool scores; a candidate review updates **only** that candidate's embedding, score, and evidence. Each video is reviewed at most once.

---

## ⚙️ Installation

**Recommended:** Python 3.11, PyTorch 2.7, CUDA 12.8 for GPU execution. The mathematical checks and feature-based training pipeline also run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.3" "numpy>=1.26" "rdflib>=7.0"
```

For raw-frame and modification-text encoding with a frozen CLIP ViT-B/16 checkpoint:

```bash
python -m pip install "transformers>=4.45" pillow
```

`FrozenCLIP` in `model.py` exposes `images(...)` and `texts(...)` methods. The feature-based pipeline accepts the resulting 512-dimensional vectors. The checkpoint is obtained through the model provider and is **not** distributed with this repository. For the documented protocol, extract twelve distinct temporal frame slots per video at CLIP's 224 × 224 input resolution and retain the identical frame-slot order for the light/deep paths. Pixel-space augmentation, if used, belongs to the upstream CLIP feature-extraction workflow.

### Source structure

```text
OEP-CVR/
├── config.py       # Reproducible model and experiment settings
├── data.py         # Manifest validation and feature storage
├── model.py        # Frozen CLIP interface and neural heads
├── semantics.py    # RDF, SPARQL, grammar, and dual target bounds
├── reasoning.py    # Conformal prediction, ranking, losses, metrics
├── scheduling.py   # Relational value estimation and review execution
├── run.py          # Train, evaluate, validate-data, and self-test
└── README.md
```

---

## 📁 Data and semantic resource contracts

### 1. Video manifest — `videos.jsonl`

Each JSON Lines record describes a temporally sampled clip. The NumPy file must contain a finite `float32` or compatible numeric array of shape **`[12, 512]`**, representing frozen CLIP frame embeddings in chronological order.

```json
{"video_id":"clip_001","feature_file":"features/clip_001.npy","label":12,"split":"train","source":"FineGym","source_video_id":"session_001","competition_id":"competition_001"}
```

- `label` is the **zero-based index** of the class in the ordered `ontology.json` vocabulary. It is required for training and calibration and can be `null` for an inference-only video.
- `split` is `train`, `validation`, `calibration`, `test`, or `gallery`. Split-sensitive training samples are selected only from the matching query partition.
- `source_video_id` and `competition_id` identify grouping keys. Reuse the same source namespace consistently for all records.
- Class indices and ontology class IDs must remain aligned across calibration, training, and inference.

### 2. Query manifest — `queries.jsonl`

```json
{"query_id":"query_001","reference_id":"clip_001","modification":"show with 2 twists","text_file":"features/query_001.npy","positives":["clip_042","clip_118"],"split":"test","source":"FineGym","source_video_id":"session_001","competition_id":"competition_001"}
```

The text feature file is a finite `[512]` CLIP text embedding. `positives` contains all officially relevant gallery IDs for the query and must exclude the reference clip itself. A query may have multiple relevant targets. The loader checks duplicate IDs, missing references and positives, feature shapes, non-finite values, and grouping conflicts across train/validation/calibration partitions.

### 3. Ontology — `ontology.json`

```json
{
  "classes": [
    {"id":"action_class_001","attributes":{"discipline":"gymnastics"},"parents":[]},
    {"id":"action_class_002","attributes":{"stage":"floor","name":"action_b","unit":"unit_a","twists":3},"parents":["action_class_001"]}
  ]
}
```

The actual experiment requires the paper's **306 sports-action classes**: 259 FineGym categories and 47 FineDiving categories. Twist counts are represented as integer **half-turns**. Omitted attributes remain unresolved. The ontology uses RDF triples and SPARQL property paths for asserted and inherited class attributes.

For ontology axioms requiring full OWL 2 entailment, first materialize the intended consequences using an OWL 2 reasoner and add `"materialized_rdf":"inferred.ttl"` to the root object. The built-in evaluator supports asserted attributes and `rdfs:subClassOf*` closure; it does not claim to replace a complete OWL 2 DL reasoner. Attribute values are treated as exclusive for the finite action-class vocabulary when checking known mismatches, while missing values remain unknown.

### 4. Frozen modification grammar — `grammar.json`

Training-derived patterns map modification text to one or more valid interpretation programs. Supported atomic operations are `set`, `negate`, `retain`, `replace`, `add`, and `remove`. Numeric `add` operations use integer half-turn counts for `twists`.

```json
{
  "mappings": {
    "show with 2 twists": [
      [{"op":"set","attribute":"twists","value":2}]
    ]
  },
  "rules": [
    {
      "pattern":"show with (?P<count>[0-9]+) twists",
      "programs":[
        [{"op":"set","attribute":"twists","value":"{count}"}]
      ]
    }
  ]
}
```

Regex matches are full-string and case-insensitive. Grammar patterns, allowable interpretations, and the action vocabulary must be finalized using the **training partition only**, before validation, calibration, or official test evaluation. Up to 16 interpretations and 12 atomic conditions per program are accepted; violations trigger the query-level fallback.

**Required external research assets:** the official TF-CoVR video/query manifests, published 306-class action and attribute definitions, training-derived grammar, the exact group split manifests, and CLIP checkpoint/embeddings are not present in the manuscript attachment and are not included in this code package. They must be supplied to reproduce the paper's numerical results. The JSON objects above specify the input contract, not the benchmark's original annotations.

---

## 🚀 Training and evaluation

### Validate input integrity

```bash
python run.py validate-data \
  --data /path/to/TF-CoVR \
  --ontology /path/to/ontology.json \
  --grammar /path/to/grammar.json
```

### Fit the representation, ranking, and value models

```bash
python run.py train \
  --data /path/to/TF-CoVR \
  --ontology /path/to/ontology.json \
  --grammar /path/to/grammar.json \
  --device cuda \
  --seed 2022 \
  --checkpoint checkpoints/oep_cvr_seed2022.pt
```

Training follows three stages and retains the lowest validation-loss checkpoint within each stage:

| Stage | Optimized components | Maximum epochs | Checkpoint selection |
|:--|:--|--:|:--|
| I. Representation | Temporal encoder, class head, video projection, query composition | 30 | Validation objective |
| II. Evidence-aware ranking | Retrieval projection and composition; frozen temporal/class predictions | 40 | Validation objective |
| III. Value estimation | `12 → 64 → 1` review-value network | 15 | Validation Huber loss |

Stage II obtains split-calibrated class sets, retains class-conflicting negatives with official negative labels, and applies the partial-order loss only to label-confirmed support-to-conflict edges. If no such negative is available, the query contributes no supervised negative ranking relation. Stage III uses official training/validation relevance only to evaluate counterfactual review gains. No test relevance labels enter optimizer updates or action selection.

Defaults include AdamW (`lr=1e-4`, `weight_decay=0.01`), query gradient accumulation to batch size 128, at most 32 evidenced negatives, temperature `0.07`, partial-order weight `0.20`, incorrect-edge penalty `2`, and patience `10`.

### Evaluate the official held-out partition

```bash
python run.py evaluate \
  --data /path/to/TF-CoVR \
  --ontology /path/to/ontology.json \
  --grammar /path/to/grammar.json \
  --device cuda \
  --checkpoint checkpoints/oep_cvr_seed2022.pt \
  --split test \
  --output results/seed2022_test.json
```

The JSON result contains `mAP@5`, `mAP@10`, `mAP@25`, `mAP@50`, `Hit@10`, and multi-target `Recall@10`, plus per-query ordered rankings, AP@50, semantic fallback status, and selected review actions. All gallery candidates are scored initially; the top 100 form the fixed review pool, while the original order of remaining candidates is retained for truncated AP evaluation. The public output uses the first ten ranked clips.

The default review costs are positive normalized units (`1.0` each). For measured cost-aware operation, supply calibrated positive costs using `--reference-cost` and `--candidate-cost`. Actual device latency, peak memory, and GMAC accounting must be profiled for the chosen hardware and cache implementation; the cost defaults are not a claim to reproduce the manuscript's measured GMAC totals.

### Published experimental protocol

The manuscript evaluates 473 official TF-CoVR test queries: **385 FineGym** and **88 FineDiving**, with five random seeds (2022–2026) and ten rounds per seed. This CLI performs one deterministic checkpoint evaluation per invocation. To follow the full protocol, run the seed/round plan under the original experimental assets and aggregate query-level statistics across runs. `reasoning.py` supplies paired cluster-bootstrap confidence intervals and Holm adjustment utilities for matched method comparisons.

| Manuscript-reported result | FineGym | FineDiving |
|:--|--:|--:|
| OEP-CVR mAP@50 | 32.96% | 54.21% |
| OEP-CVR Hit@10 | 61.17% | 77.83% |
| OEP-CVR Recall@10 | 48.95% | 69.36% |

These are **reported manuscript values**, not outcomes of the included local software checks. Exact numerical replication depends on the original semantic resources, benchmark assets, trained CLIP input features, and training protocol.

---

## 🧪 Verification and scientific safeguards

Run the integrated deterministic checks:

```bash
python run.py self-test
python -m compileall -q .
```

The checks exercise the split-conformal quantile and budget intersection, true/false/unknown semantics, partial-order precedence, supervised ranking gradients, temporal encoder dimensions, ontology subclass lookup, regex grammar, data-group integrity, cluster-bootstrap calculations, reference-wide score updates, candidate-local score updates, review limits, and query-level fallback.

**Three review cycles were completed during development:**

| Cycle | Check scope | Final finding |
|:--|:--|:--|
| 1 | Equation-level and module-level behavior | 12 checks passed; training/validation sampling was further separated |
| 2 | Semantic ranking and reference/candidate review integration | 16 checks passed; labeled-pair gain normalization and deterministic fixtures were refined |
| 3 | Split integrity, policy features, full train/evaluate execution, and regression coverage | 20 checks passed; conflict-evidenced negative selection and ontology caching were incorporated |

A separate CPU execution also traversed all three optimization stages, calibrated the class head, saved a checkpoint, reloaded it, and evaluated a small deterministic fixture. Such checks establish software behavior; they do **not** estimate statistical generalization or reproduce FineGym/FineDiving test accuracy. The manuscript's calibration-coverage discussion assumes exchangeability; conditional validity after adaptive selection is not implied by marginal split-conformal coverage.

### Scope and reproducibility notes

- **Semantic assets.** The finite grammar and ontology determine which conditions can be verified; absent attributes remain unknown. Complete OWL 2 DL consequences must be materialized externally where required.
- **Feature extraction.** RGB decoding, uniform temporal sampling, and pixel-level augmentation must reproduce the intended CLIP ViT-B/16 input distribution before feature caching. The included trainer operates on fixed frame embeddings.
- **Data partitions.** Group identifiers should be derived from source-video and competition metadata. The loader checks train/validation/calibration grouping conflicts but does not infer the official split from raw videos.
- **Runtime reporting.** Device-specific inference timing, GMACs, cache-storage measurements, and cross-method baseline adaptations require separately supplied experimental artifacts.
- **Interpretation.** Report experimental accuracy only after evaluation on the released official benchmark annotations, with independent seeds and correctly matched baselines.

---

## 📚 Citation

For manuscripts using this implementation, cite the accompanying research article by its final bibliographic record:

*Semantic Web-Driven Ontology-Constrained Compositional Video Retrieval.*

Publication venue, DOI, and repository URL should be added only after their official release. The implementation is distributed without dataset files, external pretrained weights, or a license statement asserting rights over third-party assets.
