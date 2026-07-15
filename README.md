# ⚽ FootballSearch

Semantic search over Sports Stack Exchange — ask a football (or any sport) question in natural language, get back the most relevant existing community questions, even when the wording doesn't match.

**Live demo:** 🚧 in progress — Streamlit deployment coming soon.

---

## Results

Evaluated on a held-out set of community-labeled question pairs (deduplicated before splitting to eliminate train/eval leakage). Metrics are recall@k and Mean Reciprocal Rank (MRR), computed by embedding one side of a pair as a query and checking whether the paired question appears in the top-k retrieved results.

| Model                      | recall@1 | recall@5 | recall@10 | MRR    |
|-----------------------------|:--------:|:--------:|:---------:|:------:|
| Baseline (title+body)       | 0.2792   | 0.5195   | 0.6104    | 0.3857 |
| Fine-tuned — all pairs       | **0.2922** | **0.5649** | **0.6883** | **0.4144** |
| Fine-tuned — duplicates only | 0.2792   | 0.5195   | 0.6104    | 0.3856 |
| Fine-tuned — hard negatives  | 0.2792   | 0.5195   | 0.6104    | 0.3854 |

**Headline result:** fine-tuning `all-mpnet-base-v2` with `MultipleNegativesRankingLoss` on the full set of community-labeled pairs improved recall@5 from 0.52 to 0.56 and MRR from 0.386 to 0.414 over the off-the-shelf baseline.

Both ablations — duplicates-only and hard-negative mining — came back statistically indistinguishable from baseline. This isn't a bug in either run; it's the same finding twice, from two different angles:

- **Duplicates-only** had only ~24 training pairs after the train/eval split — too little signal for the model to move at all.
- **Hard-negative mining** widened the candidate pool and produced several hundred triplets, but training on them produced no measurable change over baseline. The likely explanation is that `MultipleNegativesRankingLoss` with in-batch negatives already exposes the model to a large, effectively "hard enough" negative set at this corpus size, so explicitly mined hard negatives added little additional gradient signal. This hasn't been fully diagnosed (e.g. it's possible the mining criteria weren't selective enough, or 3 epochs wasn't enough for the harder objective to separate from baseline) — flagged here as an open question rather than a settled explanation.

Across all three experiments, **data volume — not technique — was the dominant variable.** The one configuration with enough labeled pairs (`all-pairs`, ~900+) produced the only measurable gain; every low-data or same-data-different-framing variant returned to baseline. This is the throughline of the project.

---

## Approach

**Corpus.** Sports Stack Exchange XML dump, parsed to ~6,100 questions across all tags (not football-only — football alone didn't have enough supervision pairs to fine-tune on). Question/answer bodies stripped of HTML. Community-authored duplicate and linked-question pairs (~900+) used as the supervision signal for fine-tuning.

**Retrieval.** Questions encoded with `all-mpnet-base-v2` sentence embeddings (title + body combined), indexed with FAISS (`IndexFlatIP`, inner product on normalized vectors = cosine similarity).

**Evaluation.** Pairs split 80/20, deduplicated *before* splitting to prevent bidirectional-pair leakage (Stack Exchange stores duplicate pairs symmetrically, so naive splitting can leak the same pair across train and eval). For each eval pair, one question's title+body is the query; the paired question is the target. recall@k and MRR computed over the full corpus, with a per-sport breakdown to check whether gains are football-specific or general (they generalize — improvements held across sports, not just football).

**Fine-tuning.** `sentence-transformers` `MultipleNegativesRankingLoss`, trained via a CLI script (`train.py`) with a `--base-model` flag (critical — omitting it silently falls back to a smaller/different model family) and a `--mine-from` flag controlling which pair set to mine negatives from (`duplicates_only` vs `all_pairs`). Training run on Google Colab (T4 GPU); evaluation and index-building run locally.

**Error analysis.** The clearest concrete case: a query about *"position of the defense in soccer offside"* fails to retrieve the canonical *"how is offside determined in football?"* question under the baseline model, but is correctly retrieved after fine-tuning — a good illustration of the model learning domain-specific phrasing rather than just topical similarity.

---

## Repository structure

```
footballsearch/
├── README.md
├── requirements.txt
├── data/                 # gitignored — raw dump + processed CSVs
├── notebooks/            # exploration, baseline, eval, fine-tuning, analysis
├── src/
│   ├── parse_dump.py     # XML → CSV
│   ├── build_index.py    # corpus → FAISS index
│   ├── retrieve.py       # query → top-k
│   ├── train.py          # fine-tuning CLI (--base-model, --mine-from)
│   └── evaluate.py       # recall@k + MRR computation
├── models/                # gitignored — fine-tuned model weights
├── results/
│   └── week6_metrics.json
└── app.py                 # Streamlit demo (in progress)
```

---

## Reproducibility

```bash
# setup
python -m venv footballsearch-env
source footballsearch-env/bin/activate
pip install -r requirements.txt

# data
# download sports.stackexchange.com.7z from https://archive.org/details/stackexchange
# extract into data/raw/, then:
python src/parse_dump.py

# baseline index + eval
python src/build_index.py
python src/evaluate.py --model all-mpnet-base-v2

# fine-tuning (run on Colab GPU, then download weights to models/)
python src/train.py --base-model all-mpnet-base-v2 --epochs 3 --mine-from all_pairs

# re-evaluate fine-tuned model
python src/evaluate.py --model models/finetuned-mpnet
```

---

## Limitations and future work

- Corpus is relatively small (~6,100 questions, ~900 supervision pairs) — results should be read as a controlled study of technique on limited data, not a production-scale benchmark.
- Hard-negative mining result is a genuine null finding but not yet root-caused; worth revisiting with a stricter mining threshold or more epochs before concluding the technique doesn't help here.
- No LLM-generated answer synthesis on top of retrieval (deliberately out of scope — see project rationale below).
- Streamlit demo is in progress; once live it will link here.

## Acknowledgements

- Data: [Sports Stack Exchange](https://sports.stackexchange.com/) via the [Stack Exchange data dump](https://archive.org/details/stackexchange)
- [sentence-transformers](https://www.sbert.net/) and [Hugging Face](https://huggingface.co/) for pretrained models and tooling
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
