"""
benchmark_module.py

Compare embeddings (PCA vs VAE) under KMeans, Leiden, and DBSCAN.

Notes
-----
- Silhouette and Davies-Bouldin are intrinsic scores computed in the same
  space used for clustering. They are useful but easy to over-interpret.
- ARI is reported only when reference labels exist.
- Leiden is parameterized by resolution, not by a requested cluster count.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def run_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    n_clusters = int(max(2, min(n_clusters, X.shape[0] - 1)))
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X)


def run_dbscan(X: np.ndarray, min_samples: int = 5) -> np.ndarray:
    """
    Density clustering with a per-embedding eps heuristic.

    Embeddings are standardized first so PCA and VAE scales are comparable.
    eps is the 90th percentile of the min_samples-NN distance.
    """
    if X.shape[0] < max(min_samples, 3):
        return np.zeros(X.shape[0], dtype=int)

    Xs = StandardScaler().fit_transform(X)
    k = int(min(max(min_samples, 2), Xs.shape[0]))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(Xs)
    dists = nn.kneighbors(Xs)[0][:, -1]
    eps = float(np.percentile(dists, 90))
    eps = max(eps, 1e-6)
    return DBSCAN(eps=eps, min_samples=int(min_samples)).fit_predict(Xs)


def run_leiden(adata, rep_key: str, resolution: float = 0.8, n_neighbors: int = 15) -> np.ndarray:
    import scanpy as sc

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=rep_key, n_neighbors=int(n_neighbors))
    sc.tl.leiden(adata_tmp, resolution=float(resolution), key_added="leiden")
    return adata_tmp.obs["leiden"].astype(str).values


def evaluate_clustering(X, labels, true_labels=None) -> dict:
    labels = np.asarray(labels)
    if labels.dtype.kind not in {"i", "u"}:
        # map string labels to ints; keep DBSCAN-style noise if present
        unique = [u for u in pd.unique(labels) if str(u) != "-1"]
        mapping = {str(u): i for i, u in enumerate(unique)}
        mapping["-1"] = -1
        labels = np.array([mapping.get(str(v), -1) for v in labels], dtype=int)

    valid = labels != -1
    if valid.sum() < 3 or len(set(labels[valid])) <= 1:
        return {"Silhouette": np.nan, "DB": np.nan, "ARI": np.nan, "n_clusters": int(len(set(labels[valid])))}

    Xv = np.asarray(X)[valid]
    Lv = labels[valid]
    result = {
        "Silhouette": float(silhouette_score(Xv, Lv)),
        "DB": float(davies_bouldin_score(Xv, Lv)),
        "n_clusters": int(len(set(Lv))),
        "ARI": np.nan,
    }

    if true_labels is not None:
        tl = np.asarray(true_labels)[valid]
        ok = pd.notna(pd.Series(tl)).to_numpy()
        if ok.sum() >= 3 and len(set(map(str, tl[ok]))) > 1:
            result["ARI"] = float(adjusted_rand_score(tl[ok].astype(str), Lv[ok].astype(str)))
    return result


def run_benchmark(
    adata,
    dim_methods,
    cluster_methods,
    n_clusters: int = 8,
    leiden_resolution: float = 0.8,
    true_label_col: Optional[str] = "reference_label",
    max_cells: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    results = []

    if adata.n_obs > max_cells:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()

    true_labels = None
    if true_label_col and true_label_col in adata.obs:
        true_labels = adata.obs[true_label_col].values

    for dm in dim_methods:
        rep_key = "X_pca_custom" if dm == "PCA" else "X_vae"
        if rep_key not in adata.obsm:
            results.append({
                "Embedding": dm,
                "Clustering": "—",
                "Silhouette": np.nan,
                "DB_Index": np.nan,
                "ARI": np.nan,
                "n_clusters": np.nan,
                "n_noise": np.nan,
                "status": f"missing embedding `{rep_key}`",
            })
            continue

        X = np.asarray(adata.obsm[rep_key])

        for cm in cluster_methods:
            if cm == "KMeans":
                labels = run_kmeans(X, n_clusters, random_state=random_state)
            elif cm == "DBSCAN":
                labels = run_dbscan(X)
            elif cm == "Leiden":
                labels = run_leiden(adata, rep_key, resolution=leiden_resolution)
            else:
                continue

            m = evaluate_clustering(X, labels, true_labels)
            labels_arr = np.asarray(labels).astype(str)
            n_noise = int((labels_arr == "-1").sum())
            results.append({
                "Embedding": dm,
                "Clustering": cm,
                "Silhouette": m["Silhouette"],
                "DB_Index": m["DB"],
                "ARI": m["ARI"],
                "n_clusters": m.get("n_clusters", np.nan),
                "n_noise": n_noise,
                "status": "ok",
            })

    return pd.DataFrame(results)
