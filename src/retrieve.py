import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class SemanticRetriever:
    def __init__(self, index, questions_df, model):
        self.index = index
        self.questions = questions_df
        self.model = model
        self.id_to_idx = {
            int(qid): i
            for i, qid in enumerate(self.questions['Id'])
        }

    def search(self, query: str, top_k: int = 5, exclude_id: int = None):
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores, indices = self.index.search(query_emb, top_k + 1)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.questions.iloc[idx]
            if exclude_id is not None and int(row['Id']) == exclude_id:
                continue
            results.append({
                'question_id': int(row['Id']),
                'title': row['Title'],
                'tags': row['Tags'],
                'score': float(score),
            })
            if len(results) == top_k:
                break
        
        return results