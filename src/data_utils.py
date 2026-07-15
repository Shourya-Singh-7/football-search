import pandas as pd

def combine_title_body(title: str, body: str, body_chars: int = 200) -> str:
    """
    Combines the title and body of a Stack Exchange question for embedding.
    Matches the 200-character body limit logic found in train.py and evaluate.py.
    """
    title_str = str(title).strip()
    body_str = str(body).strip()[:body_chars] if pd.notna(body) else ""
    
    return f"{title_str} {body_str}".strip()