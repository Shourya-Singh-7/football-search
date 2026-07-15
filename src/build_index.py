from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import faiss
import os

def build_index(
    questions_path='data/processed/questions.csv',
    emb_path='data/processed/baseline_embeddings.npy',
    index_path='data/processed/baseline.index',
    model_name='all-mpnet-base-v2',
    force_encode=False,
):
    questions = pd.read_csv(questions_path)

    if os.path.exists(emb_path) and not force_encode:
        print(f"Loading existing embeddings from {emb_path}")
        embeddings = np.load(emb_path).astype('float32')
        # normalise in-place (idempotent if already normalised)
        faiss.normalize_L2(embeddings)
        np.save(emb_path, embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
    else:
        print("Encoding questions...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            questions['Title'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype('float32')
        np.save(emb_path, embeddings)
        print(f"Saved embeddings: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved to {index_path}")
    return index

if __name__ == "__main__":
    build_index()