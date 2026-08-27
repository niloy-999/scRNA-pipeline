"""
Streamlit app for scRNA-seq exploration (course project).

Pipeline:
  Phase 1 → Load data + QC metrics
  Phase 2 → Filter cells, select HVGs, normalize
  Phase 3 → Train a simple VAE on the HVG matrix
  Phase 4 → Encode VAE latent space, cluster, UMAP
  Phase 5 → PCA on the same HVG matrix, cluster, UMAP
  Phase 6 → Benchmark PCA vs VAE under several clusterers
  Phase 7 → Markers on the full gene set using the selected embedding
"""

from __future__ import annotations

import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "1024"
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")



import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from benchmark_module import run_benchmark
from data_module import (
    QcThresholds,
    adata_to_dense_matrix,
    apply_qc_filtering,
    calculate_qc_metrics,
    copy_embedding,
    ensure_raw_counts_layer,
    load_10x_mtx_from_upload,
    load_h5ad_from_upload,
    load_standard_pbmc3k,
    normalize_and_select_hvgs,
    run_pca,
)
from model_module import encode_latent, get_device, train_vae


ANALYSIS_KEYS = [
    "adata_qc",
    "adata_filtered",
    "adata_hvg",
    "hvg_genes",
    "X_hvg",
    "vae_model",
    "vae_losses",
    "vae_recon_losses",
    "vae_kl_losses",
    "vae_cfg",
    "benchmark_df",
    "markers_df",
]


def init_state():
    for key in ANALYSIS_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def reset_downstream_from(phase: int) -> None:
    """Clear later-phase artifacts so stale embeddings cannot leak."""
    drop_from = {
        1: [
            "adata_filtered",
            "adata_hvg",
            "hvg_genes",
            "X_hvg",
            "vae_model",
            "vae_losses",
            "vae_recon_losses",
            "vae_kl_losses",
            "vae_cfg",
            "benchmark_df",
            "markers_df",
        ],
        2: [
            "vae_model",
            "vae_losses",
            "vae_recon_losses",
            "vae_kl_losses",
            "vae_cfg",
            "benchmark_df",
            "markers_df",
        ],
        3: ["benchmark_df", "markers_df"],
    }
    for key in drop_from.get(phase, []):
        st.session_state[key] = None


def guard(cond: bool, msg: str) -> bool:
    if not cond:
        st.warning(msg)
        return False
    return True


def _umap_figure(adata, color_key: str, title: str):
    if "X_umap" not in adata.obsm:
        raise RuntimeError("UMAP coordinates are missing.")
    plot_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    plot_df["cluster"] = adata.obs[color_key].astype(str).values
    return px.scatter(
        plot_df,
        x="UMAP1",
        y="UMAP2",
        color="cluster",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        template="plotly_dark",
    )


def _sync_embedding_to_filtered(key: str) -> None:
    if st.session_state.adata_hvg is None or st.session_state.adata_filtered is None:
        return
    if key in st.session_state.adata_hvg.obsm:
        copy_embedding(st.session_state.adata_hvg, st.session_state.adata_filtered, key)


def phase_1():
    st.header("Phase 1: Data ingestion")
    st.caption("Load PBMC3k or upload an `.h5ad` / 10x `.zip`. QC metrics are computed automatically.")

    uploaded = st.file_uploader("Upload .h5ad or .zip (10x MTX)", type=["h5ad", "zip"])

    col_a, col_b = st.columns(2)
    load_pbmc = col_a.button("Load PBMC3k", use_container_width=True)
    load_upload = col_b.button("Load uploaded file", use_container_width=True, disabled=uploaded is None)

    if load_pbmc:
        adata = load_standard_pbmc3k()
        adata = ensure_raw_counts_layer(adata)
        adata = calculate_qc_metrics(adata)
        st.session_state.adata_qc = adata
        reset_downstream_from(1)
        st.success(f"PBMC3k loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    if load_upload and uploaded is not None:
        if uploaded.name.lower().endswith(".h5ad"):
            adata = load_h5ad_from_upload(uploaded)
        else:
            adata = load_10x_mtx_from_upload(uploaded)
        adata = ensure_raw_counts_layer(adata)
        adata = calculate_qc_metrics(adata)
        st.session_state.adata_qc = adata
        reset_downstream_from(1)
        st.success(f"Data loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    if st.session_state.adata_qc is None:
        return

    adata = st.session_state.adata_qc
    if adata.uns.get("counts_warning"):
        st.warning(adata.uns["counts_warning"])

    st.subheader("AnnData summary")
    st.write(adata)

    n_ref = 0
    if "reference_label" in adata.obs:
        n_ref = int(adata.obs["reference_label"].notna().sum())
        st.caption(f"Reference labels available for {n_ref:,} / {adata.n_obs:,} cells (used later for ARI).")

    st.subheader("Expression preview (first 5 cells × 5 genes)")
    preview = adata[:5, :5]
    dense = preview.X.toarray() if hasattr(preview.X, "toarray") else preview.X
    st.dataframe(
        pd.DataFrame(dense, index=list(adata.obs_names[:5]), columns=list(adata.var_names[:5])),
        use_container_width=True,
    )

    st.subheader("Cell metadata preview")
    cols = [c for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt", "reference_label"] if c in adata.obs]
    st.dataframe(adata.obs[cols].head(8) if cols else adata.obs.head(8), use_container_width=True)


def phase_2():
    st.header("Phase 2: Filtering, normalization, HVGs")
    if not guard(st.session_state.adata_qc is not None, "Load data in Phase 1 first."):
        return

    adata_qc = st.session_state.adata_qc
    genes = adata_qc.obs["n_genes_by_counts"].to_numpy()
    pct_mt = adata_qc.obs["pct_counts_mt"].to_numpy()

    min_default = int(max(50, np.percentile(genes, 5)))
    max_default = int(min(np.percentile(genes, 99), genes.max()))
    if max_default < min_default:
        max_default = int(genes.max())
    mt_default = float(min(20.0, np.percentile(pct_mt, 99)))

    st.caption("Thresholds default to dataset percentiles. HVGs are selected on counts, then the matrix is log-normalized.")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_g = st.slider("Min genes per cell", 0, int(genes.max()), min_default)
    with c2:
        max_g = st.slider("Max genes per cell", 0, int(max(genes.max(), 1)), max_default)
    with c3:
        mt = st.slider("Max mitochondrial %", 0.0, 100.0, float(mt_default), step=0.5)

    top_n_hvgs = st.slider("Top N HVGs", 500, 5000, 2000, step=250)

    if st.button("Apply filtering + normalize + select HVGs", type="primary", use_container_width=True):
        try:
            thr = QcThresholds(min_genes=int(min_g), max_genes=int(max_g), max_pct_mt=float(mt))
            adata_f = apply_qc_filtering(adata_qc, thr)
            adata_hvg, hvgs = normalize_and_select_hvgs(adata_f, top_n_hvgs=int(top_n_hvgs))
            X = adata_to_dense_matrix(adata_hvg)

            st.session_state.adata_filtered = adata_f
            st.session_state.adata_hvg = adata_hvg
            st.session_state.hvg_genes = hvgs
            st.session_state.X_hvg = X
            reset_downstream_from(2)
            st.success(
                f"Retained {adata_f.n_obs:,} / {adata_qc.n_obs:,} cells. "
                f"HVGs: {adata_hvg.n_vars:,}. Matrix for PCA/VAE: {X.shape[0]:,} × {X.shape[1]:,}."
            )
        except Exception as e:
            st.error(str(e))

    if st.session_state.adata_filtered is not None and st.session_state.adata_hvg is not None:
        st.write(
            f"**Cells after QC:** {st.session_state.adata_filtered.n_obs:,} / {adata_qc.n_obs:,}  \n"
            f"**HVG matrix:** {st.session_state.adata_hvg.n_obs:,} cells × {st.session_state.adata_hvg.n_vars:,} genes"
        )


def phase_3():
    st.header("Phase 3: VAE training")
    st.caption(
        "Simple dense VAE with MSE reconstruction on the log-normalized HVG matrix. "
        "This is a course baseline, not scVI."
    )
    if not guard(st.session_state.X_hvg is not None, "Run Phase 2 first."):
        return

    X = st.session_state.X_hvg
    device = get_device()
    st.info(f"Training device: `{device}` · input shape: `{X.shape[0]} cells × {X.shape[1]} genes`")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        epochs = st.slider("Epochs", 5, 100, 30)
    with c2:
        latent = st.slider("Latent dim", 5, 50, 20)
    with c3:
        beta = st.slider("β (KL weight)", 0.1, 4.0, 1.0, step=0.1)
    with c4:
        lr_exp = st.select_slider("Learning rate", options=["1e-4", "5e-4", "1e-3", "2e-3"], value="1e-3")

    if st.button("Train VAE", type="primary", use_container_width=True):
        bar = st.progress(0.0)
        status = st.empty()

        def _cb(epoch, total, loss, recon, kl):
            bar.progress(epoch / total)
            status.caption(f"Epoch {epoch}/{total}  ·  loss {loss:.3f}  ·  recon {recon:.3f}  ·  KL {kl:.3f}")

        try:
            model, losses, recon_losses, kl_losses = train_vae(
                X,
                latent_dim=int(latent),
                epochs=int(epochs),
                beta=float(beta),
                lr=float(lr_exp),
                progress_cb=_cb,
            )
            st.session_state.vae_model = model
            st.session_state.vae_losses = losses
            st.session_state.vae_recon_losses = recon_losses
            st.session_state.vae_kl_losses = kl_losses
            st.session_state.vae_cfg = {
                "latent_dim": int(latent),
                "epochs": int(epochs),
                "beta": float(beta),
                "lr": float(lr_exp),
                "device": str(device),
                "seed": 7,
            }
            reset_downstream_from(3)
            if "X_vae" in getattr(st.session_state.adata_hvg, "obsm", {}):
                del st.session_state.adata_hvg.obsm["X_vae"]
            st.success(f"VAE trained on {device}. Final loss {losses[-1]:.3f}.")
        except Exception as e:
            st.error(str(e))

    if st.session_state.vae_losses is not None:
        loss_df = pd.DataFrame(
            {
                "total": st.session_state.vae_losses,
                "reconstruction": st.session_state.vae_recon_losses,
                "kl": st.session_state.vae_kl_losses,
            }
        )
        st.line_chart(loss_df)
        if st.session_state.vae_cfg:
            st.caption("Config: " + ", ".join(f"{k}={v}" for k, v in st.session_state.vae_cfg.items()))


def phase_4():
    st.header("Phase 4: VAE latent space")
    if not guard(st.session_state.vae_model is not None, "Train the VAE in Phase 3 first."):
        return
    if not guard(st.session_state.adata_hvg is not None, "Run Phase 2 first."):
        return

    resolution = st.slider("Leiden resolution (VAE)", 0.2, 2.0, 0.8, step=0.1)

    if st.button("Encode VAE + cluster + UMAP", type="primary", use_container_width=True):
        import scanpy as sc

        adata = st.session_state.adata_hvg
        X = st.session_state.X_hvg
        with st.spinner("Encoding latent space and computing UMAP..."):
            Z = encode_latent(st.session_state.vae_model, X)
            adata.obsm["X_vae"] = Z
            sc.pp.neighbors(adata, use_rep="X_vae")
            sc.tl.leiden(adata, resolution=float(resolution), key_added="vae_leiden")
            sc.tl.umap(adata)
            adata.obsm["X_umap_vae"] = adata.obsm["X_umap"].copy()
            st.session_state.adata_hvg = adata
            _sync_embedding_to_filtered("X_vae")
        st.success(f"VAE latent shape {Z.shape}. Leiden clusters: {adata.obs['vae_leiden'].nunique()}.")

    adata = st.session_state.adata_hvg
    if adata is not None and "X_umap_vae" in adata.obsm and "vae_leiden" in adata.obs:
        adata.obsm["X_umap"] = adata.obsm["X_umap_vae"]
        st.plotly_chart(_umap_figure(adata, "vae_leiden", "VAE latent UMAP"), use_container_width=True)
    elif adata is not None and "X_vae" in adata.obsm:
        st.info("VAE embedding is stored. Click the button to compute Leiden + UMAP.")


def phase_5():
    st.header("Phase 5: PCA baseline")
    st.caption("PCA is computed on the same log-normalized HVG matrix as the VAE.")
    if not guard(st.session_state.adata_hvg is not None, "Run Phase 2 first."):
        return

    c1, c2 = st.columns(2)
    with c1:
        n_comps = st.slider("PCA components", 10, 50, 50)
    with c2:
        resolution = st.slider("Leiden resolution (PCA)", 0.2, 2.0, 0.8, step=0.1)

    if st.button("Run PCA + cluster + UMAP", type="primary", use_container_width=True):
        import scanpy as sc

        adata = st.session_state.adata_hvg
        with st.spinner("Computing PCA and UMAP..."):
            if "X_pca_custom" in adata.obsm:
                del adata.obsm["X_pca_custom"]
            adata = run_pca(adata, n_comps=int(n_comps))
            sc.pp.neighbors(adata, use_rep="X_pca_custom")
            sc.tl.leiden(adata, resolution=float(resolution), key_added="pca_leiden")
            sc.tl.umap(adata)
            adata.obsm["X_umap_pca"] = adata.obsm["X_umap"].copy()
            st.session_state.adata_hvg = adata
            _sync_embedding_to_filtered("X_pca_custom")
        ev = None
        if "pca" in adata.uns and "variance_ratio" in adata.uns["pca"]:
            ev = float(np.sum(adata.uns["pca"]["variance_ratio"]))
        msg = f"PCA shape {adata.obsm['X_pca_custom'].shape}. Leiden clusters: {adata.obs['pca_leiden'].nunique()}."
        if ev is not None:
            msg += f" Variance explained: {ev:.1%}."
        st.success(msg)

    adata = st.session_state.adata_hvg
    if adata is not None and "X_umap_pca" in adata.obsm and "pca_leiden" in adata.obs:
        adata.obsm["X_umap"] = adata.obsm["X_umap_pca"]
        st.plotly_chart(_umap_figure(adata, "pca_leiden", "PCA baseline UMAP"), use_container_width=True)


def phase_6():
    st.header("Phase 6: Benchmark")
    st.caption(
        "Intrinsic scores (Silhouette, Davies–Bouldin) are computed in the same embedding used for clustering. "
        "ARI appears only when reference labels exist. Leiden uses resolution, not a fixed k."
    )
    if not guard(st.session_state.adata_hvg is not None, "Run Phase 2 first."):
        return

    adata = st.session_state.adata_hvg
    available = []
    if "X_pca_custom" in adata.obsm:
        available.append("PCA")
    if "X_vae" in adata.obsm:
        available.append("VAE")

    if not available:
        st.warning("No embeddings yet. Run Phase 4 (VAE) and/or Phase 5 (PCA).")
        return

    dims = st.multiselect("Embeddings", available, default=available)
    clus = st.multiselect("Clustering methods", ["KMeans", "Leiden", "DBSCAN"], default=["KMeans", "Leiden"])
    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("KMeans k", 2, 20, 8)
    with c2:
        leiden_res = st.slider("Leiden resolution", 0.2, 2.0, 0.8, step=0.1)

    if st.button("Run benchmark", type="primary", use_container_width=True):
        if not dims or not clus:
            st.error("Select at least one embedding and one clustering method.")
            return
        missing = [d for d in dims if (d == "PCA" and "X_pca_custom" not in adata.obsm) or (d == "VAE" and "X_vae" not in adata.obsm)]
        if missing:
            st.error("Missing embeddings: " + ", ".join(missing))
            return
        df = run_benchmark(
            adata,
            dims,
            clus,
            n_clusters=int(k),
            leiden_resolution=float(leiden_res),
            true_label_col="reference_label" if "reference_label" in adata.obs else None,
        )
        st.session_state.benchmark_df = df

    if st.session_state.benchmark_df is not None:
        df = st.session_state.benchmark_df.copy()
        st.dataframe(df, use_container_width=True)
        ok = df[df["status"] == "ok"].copy()
        if not ok.empty and ok["Silhouette"].notna().any():
            ok["combo"] = ok["Embedding"] + " + " + ok["Clustering"]
            fig = px.bar(ok, x="combo", y="Silhouette", color="Embedding", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download benchmark CSV", csv, "benchmark_results.csv", "text/csv")


def phase_7():
    st.header("Phase 7: Markers and visualization")
    if not guard(st.session_state.adata_hvg is not None, "Run Phase 2 first."):
        return
    if not guard(st.session_state.adata_filtered is not None, "Filtered full-gene object is missing."):
        return

    adata_hvg = st.session_state.adata_hvg
    adata_full = st.session_state.adata_filtered

    options = []
    if "X_pca_custom" in adata_hvg.obsm:
        options.append("PCA")
    if "X_vae" in adata_hvg.obsm:
        options.append("VAE")
    if not options:
        st.warning("Compute at least one embedding in Phase 4 or 5.")
        return

    default_ix = 0
    if st.session_state.benchmark_df is not None:
        df = st.session_state.benchmark_df
        ok = df[df["status"] == "ok"].dropna(subset=["Silhouette"])
        if not ok.empty:
            best = ok.sort_values("Silhouette", ascending=False).iloc[0]
            st.info(
                f"Highest silhouette in Phase 6: **{best['Embedding']} + {best['Clustering']}** "
                f"(silhouette={best['Silhouette']:.3f}). "
                "The plot below still uses Leiden on the chosen embedding so clusters stay graph-based and comparable."
            )
            if best["Embedding"] in options:
                default_ix = options.index(best["Embedding"])

    embedding = st.selectbox("Embedding for final Leiden + UMAP", options, index=default_ix)
    resolution = st.slider("Final Leiden resolution", 0.2, 2.0, 0.8, step=0.1)
    rep_key = "X_pca_custom" if embedding == "PCA" else "X_vae"
    _sync_embedding_to_filtered(rep_key)

    if st.button("Compute final Leiden + UMAP", type="primary", use_container_width=True):
        import scanpy as sc

        with st.spinner("Refining clusters..."):
            sc.pp.neighbors(adata_hvg, use_rep=rep_key)
            sc.tl.leiden(adata_hvg, resolution=float(resolution), key_added="final_leiden")
            sc.tl.umap(adata_hvg)
            adata_hvg.obsm["X_umap_final"] = adata_hvg.obsm["X_umap"].copy()
            adata_full.obs["final_leiden"] = adata_hvg.obs["final_leiden"].astype(str).values
            st.session_state.adata_hvg = adata_hvg
            st.session_state.adata_filtered = adata_full
            st.session_state.markers_df = None
        st.success(f"Final clusters: {adata_hvg.obs['final_leiden'].nunique()} (Leiden on {embedding}).")

    if "X_umap_final" in adata_hvg.obsm and "final_leiden" in adata_hvg.obs:
        adata_hvg.obsm["X_umap"] = adata_hvg.obsm["X_umap_final"]
        st.plotly_chart(
            _umap_figure(adata_hvg, "final_leiden", f"Final Leiden clusters ({embedding} embedding)"),
            use_container_width=True,
        )

    st.subheader("Cluster-specific marker genes")
    st.caption("Markers are ranked on the full gene set from the QC-filtered object, not only HVGs.")
    if "final_leiden" not in adata_full.obs:
        st.info("Compute final clusters first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        n_top_genes = st.slider("Top genes per cluster", 5, 100, 25)
    with c2:
        method = st.selectbox("Test", ["wilcoxon", "t-test", "logreg"], index=0)

    if st.button("Rank marker genes", type="primary", use_container_width=True):
        import scanpy as sc

        work = adata_full.copy()
        if "counts" in work.layers:
            work.X = work.layers["counts"].copy()
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        with st.spinner(f"Ranking genes with {method}..."):
            sc.tl.rank_genes_groups(
                work,
                groupby="final_leiden",
                method=method,
                n_genes=int(n_top_genes),
                use_raw=False,
            )
            df_markers = sc.get.rank_genes_groups_df(work, group=None)
            df_markers = df_markers.sort_values(["group", "pvals_adj"], ascending=[True, True])
            df_markers = df_markers.groupby("group", observed=False).head(int(n_top_genes))
            st.session_state.markers_df = df_markers

    if st.session_state.markers_df is not None:
        st.dataframe(st.session_state.markers_df, use_container_width=True, height=400)
        csv = st.session_state.markers_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download marker CSV", csv, "top_markers.csv", "text/csv", use_container_width=True)


def main():
    st.set_page_config(page_title="scRNA-seq analysis platform", layout="wide")
    init_state()
    st.title("scRNA-seq data analysis platform")
    st.caption("Course project · PCA vs a simple VAE · Scanpy + PyTorch + Streamlit")

    menu = st.sidebar.radio(
        "Pipeline",
        [
            "Phase 1: Data",
            "Phase 2: Filtering",
            "Phase 3: VAE",
            "Phase 4: Latent",
            "Phase 5: PCA",
            "Phase 6: Benchmark",
            "Phase 7: Final analysis",
        ],
    )

    if menu == "Phase 1: Data":
        phase_1()
    elif menu == "Phase 2: Filtering":
        phase_2()
    elif menu == "Phase 3: VAE":
        phase_3()
    elif menu == "Phase 4: Latent":
        phase_4()
    elif menu == "Phase 5: PCA":
        phase_5()
    elif menu == "Phase 6: Benchmark":
        phase_6()
    else:
        phase_7()


if __name__ == "__main__":
    main()
