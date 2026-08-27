"""
data_module.py

scRNA-seq ingestion and preprocessing for the Streamlit app.

Pipeline this module supports:
- Load .h5ad, 10x MTX zip, or the Scanpy PBMC3k demo
- Compute standard QC metrics
- Filter cells
- Select HVGs on raw counts, then normalize + log1p
- PCA on the same HVG matrix used by the VAE
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from anndata import AnnData


class DataModuleError(RuntimeError):
    """Raised when ingestion or preprocessing fails in a user-actionable way."""


@dataclass(frozen=True)
class QcThresholds:
    """Cell-level QC thresholds."""

    min_genes: int
    max_genes: int
    max_pct_mt: float

    def validate(self) -> None:
        if self.min_genes < 0 or self.max_genes < 0:
            raise DataModuleError("Gene-count thresholds must be non-negative.")
        if self.min_genes > self.max_genes:
            raise DataModuleError("Min genes per cell cannot exceed max genes per cell.")
        if not (0.0 <= self.max_pct_mt <= 100.0):
            raise DataModuleError("Max mitochondrial % must be between 0 and 100.")


def _import_scanpy():
    """Import Scanpy; disable Numba JIT if a macOS/Conda cache error appears."""
    try:
        import scanpy as sc  # type: ignore

        return sc
    except RuntimeError as e:
        msg = str(e)
        if "no locator available" in msg and "cannot cache function" in msg:
            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
            import scanpy as sc  # type: ignore

            return sc
        raise


def load_h5ad_from_upload(uploaded_file) -> AnnData:
    if uploaded_file is None:
        raise DataModuleError("No file provided for .h5ad loading.")

    name = getattr(uploaded_file, "name", "uploaded.h5ad")
    if not name.lower().endswith(".h5ad"):
        raise DataModuleError("Uploaded file is not an .h5ad.")

    try:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "uploaded.h5ad")
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            sc = _import_scanpy()
            adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
        return adata
    except Exception as e:  # pragma: no cover
        raise DataModuleError(f"Failed to read .h5ad: {e}") from e


def load_10x_mtx_from_upload(uploaded_file) -> AnnData:
    if uploaded_file is None:
        raise DataModuleError("No file provided for 10x .mtx loading.")

    name = getattr(uploaded_file, "name", "uploaded.zip")
    if not name.lower().endswith(".zip"):
        raise DataModuleError("For 10x MTX ingestion, please upload a .zip archive.")

    try:
        raw_bytes = uploaded_file.getbuffer()
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "tenx.zip")
            with open(zpath, "wb") as f:
                f.write(raw_bytes)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(td)

            candidate_dirs = []
            for root, _dirs, files in os.walk(td):
                files_lower = {fn.lower() for fn in files}
                if any(fn.startswith("matrix.mtx") for fn in files_lower) and (
                    any(fn.startswith("barcodes.tsv") for fn in files_lower)
                    and (
                        any(fn.startswith("features.tsv") for fn in files_lower)
                        or any(fn.startswith("genes.tsv") for fn in files_lower)
                    )
                ):
                    candidate_dirs.append(root)

            if not candidate_dirs:
                raise DataModuleError(
                    "Could not locate 10x MTX files in the archive. "
                    "Ensure it contains matrix.mtx(+.gz), barcodes.tsv(+.gz), "
                    "and features.tsv/genes.tsv (+.gz)."
                )

            tenx_dir = sorted(candidate_dirs, key=len)[0]
            sc = _import_scanpy()
            adata = sc.read_10x_mtx(tenx_dir, var_names="gene_symbols", cache=False)
            adata.var_names_make_unique()
            return adata
    except DataModuleError:
        raise
    except Exception as e:  # pragma: no cover
        raise DataModuleError(f"Failed to read 10x MTX zip: {e}") from e


def _attach_pbmc3k_reference_labels(adata: AnnData) -> AnnData:
    """
    Attach Scanpy's processed PBMC3k cluster labels where barcodes overlap.

    These are published Louvain clusters from the processed tutorial object,
    not a gold-standard expert atlas. Use them only as a weak ARI reference.
    """
    try:
        sc = _import_scanpy()
        ref = sc.datasets.pbmc3k_processed()
        if "louvain" not in ref.obs.columns:
            return adata
        common = adata.obs_names.intersection(ref.obs_names)
        adata.obs["reference_label"] = pd_na_string(adata.n_obs)
        if len(common) > 0:
            adata.obs.loc[common, "reference_label"] = (
                ref.obs.loc[common, "louvain"].astype(str).values
            )
    except Exception:
        pass
    return adata


def pd_na_string(n: int) -> np.ndarray:
    return np.array([None] * n, dtype=object)


def load_standard_pbmc3k() -> AnnData:
    """Load Scanpy's built-in PBMC3k counts and attach reference labels when possible."""
    try:
        sc = _import_scanpy()
        adata = sc.datasets.pbmc3k()
        adata.var_names_make_unique()
        adata.uns["dataset_name"] = "pbmc3k"
        return _attach_pbmc3k_reference_labels(adata)
    except Exception as e:  # pragma: no cover
        raise DataModuleError(f"Failed to load pbmc3k: {e}") from e


def _looks_like_counts(X) -> bool:
    """Heuristic: integer-valued non-negative matrix is treated as raw counts."""
    sample = X[: min(200, X.shape[0])]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    sample = np.asarray(sample)
    if sample.size == 0:
        return True
    if np.any(sample < 0):
        return False
    return bool(np.allclose(sample, np.round(sample), atol=1e-6))


def ensure_raw_counts_layer(adata: AnnData, layer_name: str = "counts") -> AnnData:
    """
    Ensure a raw counts layer exists.

    If `.X` already looks like counts and no counts layer exists, copy `.X`.
    If `.X` looks normalized, keep whatever is in `.layers['counts']` if present;
    otherwise copy `.X` and flag the object so QC can warn the user.
    """
    if layer_name in adata.layers:
        return adata

    looks_counts = _looks_like_counts(adata.X)
    adata.layers[layer_name] = adata.X.copy()
    adata.uns["counts_layer_is_guess"] = not looks_counts
    if not looks_counts:
        adata.uns["counts_warning"] = (
            "`.X` does not look like raw UMI counts. QC / seurat_v3 HVG "
            "selection may be unreliable. Prefer files with a counts layer."
        )
    return adata


def calculate_qc_metrics(adata: AnnData, mt_gene_prefix: str = "MT-") -> AnnData:
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise DataModuleError("Empty AnnData matrix; cannot compute QC metrics.")

    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith(mt_gene_prefix.upper())

    sc = _import_scanpy()
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    return adata


def apply_qc_filtering(
    adata: AnnData,
    thresholds: QcThresholds,
    *,
    counts_layer: Optional[str] = "counts",
) -> AnnData:
    thresholds.validate()

    for col in ("n_genes_by_counts", "total_counts", "pct_counts_mt"):
        if col not in adata.obs.columns:
            raise DataModuleError(f"Missing QC metric `{col}`. Run calculate_qc_metrics first.")

    mask = (
        (adata.obs["n_genes_by_counts"] >= thresholds.min_genes)
        & (adata.obs["n_genes_by_counts"] <= thresholds.max_genes)
        & (adata.obs["pct_counts_mt"] <= thresholds.max_pct_mt)
    )
    filtered = adata[mask].copy()
    if filtered.n_obs == 0:
        raise DataModuleError("QC filtering removed all cells. Relax thresholds and try again.")
    filtered.uns["qc_thresholds"] = {
        "min_genes": thresholds.min_genes,
        "max_genes": thresholds.max_genes,
        "max_pct_mt": thresholds.max_pct_mt,
    }
    filtered.uns["qc_cells_before"] = int(adata.n_obs)
    filtered.uns["qc_cells_after"] = int(filtered.n_obs)
    return filtered


def normalize_and_select_hvgs(
    adata: AnnData,
    *,
    top_n_hvgs: int = 2000,
    target_sum: float = 1e4,
    flavor: str = "seurat_v3",
) -> Tuple[AnnData, np.ndarray]:
    """
    Select HVGs on counts, subset, then normalize + log1p.

    seurat_v3 expects discrete counts. We therefore run HVG selection first
    (using layers['counts'] when present) and only then normalize.
    """
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise DataModuleError("Empty AnnData matrix; cannot normalize/select HVGs.")

    adata_proc = adata.copy()
    sc = _import_scanpy()

    n_top = int(min(top_n_hvgs, adata_proc.n_vars))
    hvg_kw: dict = {
        "n_top_genes": n_top,
        "flavor": flavor,
        "subset": True,
        "inplace": True,
    }
    if flavor == "seurat_v3" and "counts" in adata_proc.layers:
        hvg_kw["layer"] = "counts"

    sc.pp.highly_variable_genes(adata_proc, **hvg_kw)

    if adata_proc.n_vars == 0:
        raise DataModuleError("HVG selection produced 0 genes. Reduce top_n_hvgs or check input.")

    sc.pp.normalize_total(adata_proc, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata_proc)
    adata_proc.uns["hvg_flavor"] = flavor
    adata_proc.uns["n_hvgs"] = int(adata_proc.n_vars)
    return adata_proc, adata_proc.var_names.to_numpy()


def adata_to_dense_matrix(adata: AnnData) -> np.ndarray:
    """Convert AnnData `.X` to a dense float32 array (cells x genes)."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def run_pca(
    adata: AnnData,
    n_comps: int = 50,
    scale: bool = True,
    key: str = "X_pca_custom",
    random_state: int = 42,
) -> AnnData:
    """
    PCA on the current `.X` of `adata`.

    Call this on the HVG log-normalized object so PCA and the VAE see
    the same feature matrix.
    """
    sc = _import_scanpy()
    adata_tmp = adata.copy()
    if scale:
        sc.pp.scale(adata_tmp, max_value=10)

    max_comps = max(1, min(int(n_comps), adata_tmp.n_obs - 1, adata_tmp.n_vars))
    sc.tl.pca(adata_tmp, n_comps=max_comps, random_state=random_state)
    adata.obsm[key] = np.asarray(adata_tmp.obsm["X_pca"], dtype=np.float32)
    if "pca" in adata_tmp.uns:
        adata.uns["pca"] = adata_tmp.uns["pca"]
    return adata


def copy_embedding(src: AnnData, dst: AnnData, key: str) -> None:
    """Copy an obsm embedding between objects that share cell barcodes."""
    if key not in src.obsm:
        raise DataModuleError(f"Embedding `{key}` is missing.")
    if src.n_obs == dst.n_obs and np.array_equal(src.obs_names.values, dst.obs_names.values):
        dst.obsm[key] = np.asarray(src.obsm[key])
        return
    common = dst.obs_names.intersection(src.obs_names)
    if len(common) != dst.n_obs:
        raise DataModuleError(
            f"Cannot copy `{key}`: cell barcodes do not align "
            f"({len(common)} shared / {dst.n_obs} destination cells)."
        )
    indexer = src.obs_names.get_indexer(dst.obs_names)
    dst.obsm[key] = np.asarray(src.obsm[key])[indexer]
