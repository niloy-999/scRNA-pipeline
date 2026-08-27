# scRNA-seq Data Analysis Platform

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22132422.svg)](https://doi.org/10.5281/zenodo.22132422)

Streamlit's interactive application for single-cell RNA-seq analysis: quality control, highly variable gene selection, PCA and variational autoencoder (VAE) embeddings, clustering, benchmarking, and marker-gene export.

It was developed using scikit-learn, PyTorch, and Scanpy. It is designed to support local exploratory analysis of PBMC-scale and other modest `.h5ad` / 10x datasets. It is not a substitute for Scanpy, Seurat, or scvi-tools in atlas-scale or multi-batch investigations.

---

## Features

- Load `.h5ad`, a 10x Genomics MTX zip, or Scanpy’s PBMC3k demo dataset
- Cell-level QC (gene counts, mitochondrial fraction) with dataset-aware defaults
- HVG selection on raw counts (`seurat_v3`), then library-size normalization and log1p
- PCA and a configurable dense VAE on the **same** HVG matrix
- Leiden, KMeans, and DBSCAN on either embedding
- Benchmark table (Silhouette, Davies–Bouldin, ARI when labels exist) with CSV export
- Marker genes on the full gene set, with CSV export
- Upload limit configured to 1024 MB (see `.streamlit/config.toml` and `run.py`)

---

## Installation

Python 3.10+ is recommended. A GPU (CUDA or Apple MPS) is optional.

```bash
git clone https://github.com/niloy-999/scRNA-pipeline.git
cd scRNA-pipeline

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install torch` does not pick up CUDA, use the command from [pytorch.org](https://pytorch.org) for your platform.

---

## Usage

Preferred launch (sets a 1024 MB upload limit before Streamlit starts):

```bash
python run.py
```

Alternative:

```bash
streamlit run app.py --server.maxUploadSize=1024 --server.maxMessageSize=1024
```

Open the local URL Streamlit prints (typically `http://localhost:8501`).

**Suggested workflow**

1. Phase 1 — load PBMC3k or upload a counts `.h5ad` / 10x zip  
2. Phase 2 — filter cells, normalize, select HVGs  
3. Phase 3 — train the VAE  
4. Phase 4 — encode the VAE latent space, Leiden, UMAP  
5. Phase 5 — PCA on the same HVG matrix, Leiden, UMAP  
6. Phase 6 — benchmark embeddings and clusterers; download CSV  
7. Phase 7 — final Leiden view and marker-gene table; download CSV  

Prior to the completion of the VAE or PCA, Phase 2 must be completed. The HVG log-normalized matrix that is generated is used to calculate both embeddings.

Do not commit large `.h5ad` files. PBMC3k is downloaded by Scanpy on first use.

---

## Repository layout

```
.
├── app.py                 # Streamlit UI
├── run.py                 # Launcher with 1024 MB upload limit
├── data_module.py         # IO, QC, HVG, PCA
├── model_module.py        # VAE definition and training
├── benchmark_module.py    # Clustering and metrics
├── requirements.txt
└── README.md
```

---

## Methods

**Quality control.** Cells are retained if

```text
min_genes ≤ n_genes ≤ max_genes  and  pct_mitochondrial ≤ max_pct_mt
```

Slider defaults follow dataset percentiles rather than a single hardcoded cut.

**HVGs.** `scanpy.pp.highly_variable_genes(..., flavor="seurat_v3")` is run on the `counts` layer before normalization. The object is subset to those genes, total-count normalized to 10,000, and log1p-transformed.

**PCA.** Computed on that HVG matrix after `scale(max_value=10)`. Stored as `obsm["X_pca_custom"]`.

**VAE.** Two-layer encoder/decoder (hidden size 256, ReLU, dropout 0.1). Loss is per-cell MSE plus β · KL to N(0, I) on already log-normalized expression. Defaults: 30 epochs, latent size 20, β = 1, Adam `1e-3`, gradient clip 1.0, seed 7. The stored embedding is the posterior mean. This is a compact baseline VAE, not a count-based model such as scVI.

**Clustering.**

- KMeans: user-chosen `k`
- Leiden: user-chosen resolution (not a cluster count)
- DBSCAN: embeddings are standardized; `eps` is the 90th percentile of the k-NN distance

**Metrics.** Silhouette and Davies–Bouldin in the embedding used for clustering. Adjusted Rand Index is reported when `reference_label` is present (for PBMC3k, overlapping barcodes from `scanpy.datasets.pbmc3k_processed()`). Those labels are tutorial clusters, not an official atlas annotation.

**Markers.** Ranked according to the complete gene set of the QC-filtered object. In order to prevent the HVG log matrix from being inadvertently utilized, the counts are re-normalized during this stage.

---

## Reproducibility

| Setting | Value |
|---|---|
| VAE seed | 7 |
| PCA `random_state` | 42 |
| KMeans `random_state` | 42 |
| Leiden | Scanpy / igraph when available; resolution is set in the UI |

GPU/MPS runs are not guaranteed bit-identical across machines. Record the device string shown in Phase 3 when comparing training curves.

Phase 6 and Phase 7 CSV exports are the citable outputs of a run.

---

## Limitations

- Validated primarily on PBMC3k (~2.7k cells). Benchmarking subsamples above 50,000 cells.
- The built-in VAE uses MSE on log-normalized values. It does not model UMI counts, library size, or batches.
- Silhouette on the same space used for clustering should not be the only decision criterion. Prefer ARI against labels and visual inspection.
- No batch correction (Harmony, scVI, Scanorama) in the current release.
- Uploaded matrices that are already log-normalized will make QC and `seurat_v3` HVG selection unreliable. Prefer raw counts.
- Mitochondrial genes are detected with a case-insensitive `MT-` / `mt-` prefix.

---

## Future work

Planned in roughly this order:

1. **scVI baseline** — add a count-based VAE (scvi-tools) next to PCA and the simple VAE so the deep-learning comparison matches current practice.
2. **More labeled datasets** — pancreas and at least one multi-sample or perturbation set, with ARI/NMI against published cell-type labels.
3. **Batch correction** — Harmony and/or scVI batch keys for multi-library uploads.
4. **Export pack** — one download containing `.h5ad`, `params.json` (thresholds, seeds, package versions), benchmark CSV, and markers CSV.
5. **Packaging** — `pip` install, Docker image, and an optional hosted demo limited to PBMC3k.
6. **Tests and CI** — unit tests for QC, HVG count, embedding shapes, and benchmark columns; GitHub Actions on supported Python versions.
7. **Automated annotation** — optional mapping of clusters to a reference atlas after the embedding is fixed.
8. **Trajectory / spatial** — pseudotime and spatial assay support only after the core embedding comparison is stable.

These items are the path from a working local app to a version that can be cited and extended.

---

## Citation

If you use this software, please cite:

Tasnim Ul Islam. (2026). scRNA-pipeline: scRNA-seq data analysis platform (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.22132422

---

## License

Add an OSI-approved license before the first public push (MIT or BSD-3-Clause is recommended).
