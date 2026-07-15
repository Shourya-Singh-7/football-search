from src.retrieve import SemanticRetriever

r = SemanticRetriever(
    index_path='data/processed/baseline.index',
    embeddings_path='data/processed/baseline_embeddings.npy',
    questions_path='data/processed/questions.csv',
    model_name='all-mpnet-base-v2',
)

for res in r.search("What is a false 9?", top_k=5):
    print(f"[{res['score']:.3f}] {res['title']}")
