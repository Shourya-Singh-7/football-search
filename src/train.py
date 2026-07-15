"""
train.py
--------
Fine-tuning pipeline for FootballSearch.

THREE MODEL VARIANTS:
    1. ft-all-pairs        — all 620 train pairs, MultipleNegativesRankingLoss
    2. ft-duplicates-only  — ~134 duplicate pairs only, same loss (ablation: pair quality)
    3. ft-hard-negatives   — all pairs + mined hard negatives, TripletLoss (ablation: hard neg)

WORKFLOW:
    Step A (run locally, CPU fine):
        python src/train.py --mode mine_negatives
        → writes data/processed/hard_negatives.csv

    Step B (run on Colab with T4 GPU):
        python src/train.py --mode train --variant all_pairs
        python src/train.py --mode train --variant duplicates_only
        python src/train.py --mode train --variant hard_negatives

    Upload to Colab:  data/processed/train_pairs.csv
                      data/processed/questions.csv
                      data/processed/hard_negatives.csv   (for variant 3)
    Download from Colab: models/ft-all-pairs/
                         models/ft-duplicates-only/
                         models/ft-hard-negatives/

ENCODING CONSISTENCY:
    Training pairs are encoded with combine_title_body() — the same function
    used in build_index.py and evaluate.py — so that fine-tuned model comparisons
    against the title+body dense baseline are fair.
"""

import argparse
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def combine_title_body(row: pd.Series, body_chars: int = 200) -> str:
    """
    Must match the identical function in build_index.py and evaluate.py.
    All encoding in this project goes through this function.
    """
    title = str(row["Title"]).strip()
    body = str(row["Body"]).strip()[:body_chars]
    return f"{title} {body}".strip()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load questions, train pairs, and (if present) hard negatives."""
    processed = Path("data/processed")
    questions = pd.read_csv(processed / "questions.csv")
    train_pairs = pd.read_csv(processed / "train_pairs.csv")

    hard_neg_path = processed / "hard_negatives.csv"
    hard_negatives = (
        pd.read_csv(hard_neg_path) if hard_neg_path.exists() else pd.DataFrame()
    )
    return questions, train_pairs, hard_negatives


def get_question_text(qid: int, questions_df: pd.DataFrame, id_map: dict) -> str | None:
    """Return combine_title_body text for a question ID, or None if not found."""
    if qid not in id_map:
        return None
    return combine_title_body(questions_df.iloc[id_map[qid]])


# ---------------------------------------------------------------------------
# Step A — Hard negative mining  (run locally, CPU)
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    questions: pd.DataFrame,
    train_pairs: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    top_n: int = 20,
    output_path: str = "data/processed/hard_negatives.csv",
) -> pd.DataFrame:
    """
    For each DUPLICATE train pair (A, B):
      1. Encode A's title+body.
      2. Retrieve top-N from the corpus using the pre-trained baseline model.
      3. Remove B and any question that is linked to A in any pair.
      4. Take the highest-ranked remaining question as the hard negative for A.

    Why duplicate pairs only for mining?
    Duplicate pairs (LinkTypeId=3) are high-confidence positives. Linked pairs
    (LinkTypeId=1) are noisier — using them as anchors risks mining false negatives
    (questions that are actually relevant but not labelled as such).

    Writes data/processed/hard_negatives.csv with columns:
        anchor_id, positive_id, negative_id
    """
    print("Mining hard negatives …")
    print(f"  Base model  : {model_name}")
    print(f"  Retrieve top: {top_n}")

    id_map = {int(qid): i for i, qid in enumerate(questions["Id"])}

    # Build a set of all question IDs known to be related to each question
    # (both directions, all link types) so we don't mine them as negatives.
    related: dict[int, set] = {}
    for _, row in train_pairs.iterrows():
        a, b = int(row["PostId"]), int(row["RelatedPostId"])
        related.setdefault(a, set()).add(b)
        related.setdefault(b, set()).add(a)

    # Encode the full corpus once
    print("  Encoding corpus …")
    model = SentenceTransformer(model_name)
    texts = [combine_title_body(row) for _, row in questions.iterrows()]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Build a temporary FAISS index for mining
    vecs = embeddings.copy()
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Mine: duplicate pairs only
    dup_pairs = train_pairs[train_pairs["LinkTypeId"] == 3]
    print(f"  Duplicate train pairs to mine from: {len(dup_pairs)}")

    records = []
    skipped = 0

    for _, row in dup_pairs.iterrows():
        anchor_id = int(row["PostId"])
        positive_id = int(row["RelatedPostId"])

        if anchor_id not in id_map:
            skipped += 1
            continue

        # Encode anchor
        anchor_text = combine_title_body(questions.iloc[id_map[anchor_id]])
        anchor_emb = model.encode([anchor_text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(anchor_emb)

        # Retrieve top_n + buffer (we'll filter several out)
        scores, indices = index.search(anchor_emb, top_n + 10)

        # Exclude: the anchor itself, the positive, and anything related to anchor
        excluded = {anchor_id, positive_id} | related.get(anchor_id, set())

        hard_neg_id = None
        for idx in indices[0]:
            if idx == -1:
                continue
            candidate_id = int(questions.iloc[idx]["Id"])
            if candidate_id not in excluded:
                hard_neg_id = candidate_id
                break

        if hard_neg_id is None:
            skipped += 1
            continue

        records.append({
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": hard_neg_id,
        })

    result = pd.DataFrame(records)
    result.to_csv(output_path, index=False)
    print(f"  Mined {len(result)} hard negatives  ({skipped} skipped)")
    print(f"  Saved → {output_path}")
    return result


# ---------------------------------------------------------------------------
# Step B — Training  (run on Colab with T4 GPU)
# ---------------------------------------------------------------------------

def build_all_pairs_examples(
    questions: pd.DataFrame,
    train_pairs: pd.DataFrame,
) -> list[InputExample]:
    """
    All 620 train pairs → InputExample(texts=[anchor_text, positive_text]).
    Used with MultipleNegativesRankingLoss (in-batch negatives, no explicit negatives needed).
    """
    id_map = {int(qid): i for i, qid in enumerate(questions["Id"])}
    examples = []

    for _, row in train_pairs.iterrows():
        a_id, b_id = int(row["PostId"]), int(row["RelatedPostId"])
        a_text = get_question_text(a_id, questions, id_map)
        b_text = get_question_text(b_id, questions, id_map)
        if a_text is None or b_text is None:
            continue
        examples.append(InputExample(texts=[a_text, b_text]))

    print(f"  all-pairs examples: {len(examples)}")
    return examples


def build_duplicates_only_examples(
    questions: pd.DataFrame,
    train_pairs: pd.DataFrame,
) -> list[InputExample]:
    """
    Duplicate pairs only (LinkTypeId == 3) → InputExample(texts=[a, b]).
    Ablation: does higher-quality but smaller training data outperform all-pairs?
    """
    id_map = {int(qid): i for i, qid in enumerate(questions["Id"])}
    dup_pairs = train_pairs[train_pairs["LinkTypeId"] == 3]
    examples = []

    for _, row in dup_pairs.iterrows():
        a_id, b_id = int(row["PostId"]), int(row["RelatedPostId"])
        a_text = get_question_text(a_id, questions, id_map)
        b_text = get_question_text(b_id, questions, id_map)
        if a_text is None or b_text is None:
            continue
        examples.append(InputExample(texts=[a_text, b_text]))

    print(f"  duplicates-only examples: {len(examples)}")
    return examples


def build_hard_negative_examples(
    questions: pd.DataFrame,
    hard_negatives: pd.DataFrame,
) -> list[InputExample]:
    """
    Mined triplets → InputExample(texts=[anchor, positive, hard_negative]).
    Used with TripletLoss.
    """
    if hard_negatives.empty:
        raise FileNotFoundError(
            "data/processed/hard_negatives.csv not found. "
            "Run: python src/train.py --mode mine_negatives"
        )

    id_map = {int(qid): i for i, qid in enumerate(questions["Id"])}
    examples = []

    for _, row in hard_negatives.iterrows():
        a = get_question_text(int(row["anchor_id"]), questions, id_map)
        p = get_question_text(int(row["positive_id"]), questions, id_map)
        n = get_question_text(int(row["negative_id"]), questions, id_map)
        if None in (a, p, n):
            continue
        examples.append(InputExample(texts=[a, p, n]))

    print(f"  hard-negative triplet examples: {len(examples)}")
    return examples


def train(
    variant: str,
    questions: pd.DataFrame,
    train_pairs: pd.DataFrame,
    hard_negatives: pd.DataFrame,
    base_model: str = "all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 100,
):
    """
    Train one of the three model variants and save to models/<variant>/.

    variant choices:
        "all_pairs"       → MultipleNegativesRankingLoss, all 620 pairs
        "duplicates_only" → MultipleNegativesRankingLoss, ~134 duplicate pairs
        "hard_negatives"  → TripletLoss, mined triplets
    """
    output_dir = f"models/ft-{variant.replace('_', '-')}"
    print(f"\n{'='*60}")
    print(f"  Training variant : {variant}")
    print(f"  Base model       : {base_model}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Output           : {output_dir}")
    print(f"{'='*60}\n")

    model = SentenceTransformer(base_model)

    if variant == "all_pairs":
        examples = build_all_pairs_examples(questions, train_pairs)
        dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = losses.MultipleNegativesRankingLoss(model=model)

    elif variant == "duplicates_only":
        examples = build_duplicates_only_examples(questions, train_pairs)
        dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = losses.MultipleNegativesRankingLoss(model=model)

    elif variant == "hard_negatives":
        examples = build_hard_negative_examples(questions, hard_negatives)
        dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss = losses.TripletLoss(model=model)

    else:
        raise ValueError(f"Unknown variant: {variant!r}. Choose: all_pairs, duplicates_only, hard_negatives")

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_dir,
        optimizer_params={'lr': 1e-5}, # <-- Added a smaller learning rate
    )
    print(f"\nSaved fine-tuned model → {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FootballSearch training pipeline")
    parser.add_argument(
        "--mode",
        choices=["mine_negatives", "train"],
        required=True,
        help="mine_negatives: run locally before Colab. train: run on Colab.",
    )
    parser.add_argument(
        "--variant",
        choices=["all_pairs", "duplicates_only", "hard_negatives"],
        default="all_pairs",
        help="Which training configuration to run (only used with --mode train).",
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    questions, train_pairs, hard_negatives = load_data()

    if args.mode == "mine_negatives":
        mine_hard_negatives(
            questions=questions,
            train_pairs=train_pairs,
            model_name=args.base_model,
        )

    elif args.mode == "train":
        train(
            variant=args.variant,
            questions=questions,
            train_pairs=train_pairs,
            hard_negatives=hard_negatives,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()