import json
import numpy as np
import pandas as pd
from collections import defaultdict


def make_eval_splits(pairs_df, train_ratio=0.8, random_seed=42):
    
    # Split pairs into train and eval sets
    pairs_df = pairs_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(pairs_df) * train_ratio)
    train_df = pairs_df.iloc[:split_idx].reset_index(drop=True)
    eval_df  = pairs_df.iloc[split_idx:].reset_index(drop=True)
    return train_df, eval_df

def combine_title_body(row: pd.Series, body_chars: int = 200) -> str:
    title = str(row["Title"]).strip()
    body = str(row["Body"]).strip()[:body_chars]
    return f"{title} {body}".strip()

def evaluate_retrieval(retriever, eval_pairs, questions_df, k_values=[1, 5, 10], max_k=10):
    """
    Computes recall@k and MRR over the eval pairs.

    retriever   : object with a .search(query, top_k) method
                  each result must have a 'question_id' field
    eval_pairs  : list of (post_id, related_post_id, link_type_id) tuples
    questions_df: DataFrame with columns [Id, Title, ...]
    k_values    : list of k values to compute recall@k for
    max_k       : how deep to retrieve (should be >= max(k_values))

    Returns dict with recall@1, recall@5, recall@10, mrr
    """
    
    question_id_to_idx = {
        int(qid): i for i, qid in enumerate(questions_df['Id'])
    }

    hits    = defaultdict(int)
    rr_sum  = 0.0
    total   = 0

    for post_id, related_post_id, _ in eval_pairs:
        post_id         = int(post_id)
        related_post_id = int(related_post_id)

        # Skip if either question isn't in the corpus
        if post_id not in question_id_to_idx or related_post_id not in question_id_to_idx:
            continue

        query_title = combine_title_body(questions_df.iloc[question_id_to_idx[post_id]])

        # Retrieve max_k + 1 so we have room after excluding the query itself
        results    = retriever.search(query_title, top_k=max_k + 1)
        result_ids = [r['question_id'] for r in results if r['question_id'] != post_id]

        # recall@k
        for k in k_values:
            if related_post_id in result_ids[:k]:
                hits[k] += 1

        # MRR — reciprocal rank of the first correct result
        for rank, qid in enumerate(result_ids[:max_k], start=1):
            if qid == related_post_id:
                rr_sum += 1.0 / rank
                break

        total += 1

    if total == 0:
        raise ValueError("No valid eval pairs found — check question IDs in pairs file.")

    metrics = {f'recall@{k}': round(hits[k] / total, 4) for k in k_values}
    metrics['mrr']   = round(rr_sum / total, 4)
    metrics['total'] = total
    return metrics


def evaluate_by_sport(retriever, eval_pairs, questions_df, k=5):
    """
    Computes recall@k broken down by the dominant sport tag of each pair.
    Sport is determined by the tags on the query question (post_id side).

    Returns a dict: {sport_tag: {'recall@k': float, 'count': int}}
    """

    # Build a tag lookup: question Id -> list of tags
    def parse_tags(tag_str):
        if not isinstance(tag_str, str):
            return []
        # Tags are stored as <tag1><tag2> in Stack Exchange dumps
        return [t for t in tag_str.strip('|').split('|') if t]
    id_to_tags = {
        int(row['Id']): parse_tags(row['Tags'])
        for _, row in questions_df.iterrows()
    }
    question_id_to_idx = {
        int(qid): i for i, qid in enumerate(questions_df['Id'])
    }

    sport_hits  = defaultdict(int)
    sport_total = defaultdict(int)

    for post_id, related_post_id, _ in eval_pairs:
        post_id         = int(post_id)
        related_post_id = int(related_post_id)

        if post_id not in question_id_to_idx or related_post_id not in question_id_to_idx:
            continue

        tags = id_to_tags.get(post_id, [])
        # Use the first tag as the dominant sport; fall back to 'untagged'
        sport = tags[0] if tags else 'untagged'

        query_title = combine_title_body(questions_df.iloc[question_id_to_idx[post_id]])
        results     = retriever.search(query_title, top_k=k + 1)
        result_ids  = [r['question_id'] for r in results if r['question_id'] != post_id]

        sport_total[sport] += 1
        if related_post_id in result_ids[:k]:
            sport_hits[sport] += 1

    breakdown = {}
    for sport, count in sorted(sport_total.items(), key=lambda x: -x[1]):
        breakdown[sport] = {
            f'recall@{k}': round(sport_hits[sport] / count, 4),
            'n_pairs': count,
        }
    return breakdown


def save_metrics(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")