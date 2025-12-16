import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_quality_indicator(
    df_meta,
    embedding_file,
    key_col='pid',
    industry_col='startup_industry_1',
    acceptance_col='acceptance_rate',
    top_ratio=0.05,
    file_type='csv'  # or 'npz'
):
    """
    Compute multimodal quality indicator based on seed BPs (top % by acceptance rate per industry).

    Parameters:
    - df_meta (pd.DataFrame): metadata with columns [key_col, industry_col, acceptance_col]
    - embedding_file (str): path to embedding file (.csv or .npz)
    - key_col (str): column name for BP identifier (e.g., 'pid')
    - industry_col (str): column for industry grouping
    - acceptance_col (str): column for acceptance rate (used to rank seed BPs)
    - top_ratio (float): proportion of top BPs per industry to use as seeds (e.g., 0.05 for top 5%)
    - file_type (str): 'csv' or 'npz'

    Returns:
    - pd.DataFrame: with columns [key_col, 'quality_score']
    """
    # Step 1: Load embeddings
    if file_type == 'csv':
        emb_df = pd.read_csv(embedding_file)
        # Assume first column is key (e.g., pid), rest are embeddings
        key_list = emb_df.iloc[:, 0].astype(str).tolist()
        embeddings = emb_df.iloc[:, 1:].values
        emb_dict = {k: embeddings[i] for i, k in enumerate(key_list)}
    elif file_type == 'npz':
        data = np.load(embedding_file)
        emb_dict = {k: data[k] for k in data.files}
        # Ensure values are 1D arrays (average over pages if needed)
        emb_dict = {k: v if v.ndim == 1 else np.mean(v, axis=0) for k, v in emb_dict.items()}
    else:
        raise ValueError("file_type must be 'csv' or 'npz'")

    # Step 2: Keep only BPs that have both metadata and embedding
    df_meta = df_meta.copy()
    df_meta[key_col] = df_meta[key_col].astype(str)
    valid_pids = set(df_meta[key_col]) & set(emb_dict.keys())
    df_valid = df_meta[df_meta[key_col].isin(valid_pids)].copy()

    # Step 3: Group by industry and compute quality
    results = []

    for industry, group in df_valid.groupby(industry_col):

        # Sort by acceptance rate descending
        group = group.sort_values(by=acceptance_col, ascending=False)
        n_seed = max(1, int(len(group) * top_ratio))
        seed_pids = group.head(n_seed)[key_col].tolist()
        candidate_pids = group[key_col].tolist()

        # Get embeddings
        try:
            seed_embs = np.stack([emb_dict[pid] for pid in seed_pids])
            cand_embs = np.stack([emb_dict[pid] for pid in candidate_pids])
        except Exception as e:
            print(f"Skipping industry {industry}: embedding error - {e}")
            continue

        # Compute avg cosine similarity to seed BPs
        sim_matrix = cosine_similarity(cand_embs, seed_embs)  # shape: (n_cand, n_seed)
        avg_sim = sim_matrix.mean(axis=1)  # average over seeds

        res = pd.DataFrame({
            key_col: candidate_pids,
            'quality_score': avg_sim
        })
        results.append(res)

    if not results:
        raise ValueError("No valid industry groups found.")

    return pd.concat(results, ignore_index=True)


df_meta = pd.read_csv('metadata.csv')  # columns: pid, startup_industry_1, acceptance_rate

vision_scores = compute_quality_indicator(
    df_meta=df_meta,
    embedding_file='./data/dit_visual_features.npz',
    key_col='pid',
    industry_col='startup_industry_1',
    acceptance_col='acceptance_rate',
    top_ratio=0.05,
    file_type='npz'
)
vision_scores.to_csv('vision_quality.csv', index=False)