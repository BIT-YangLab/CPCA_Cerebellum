"""
Merged correlation analysis for cerebellum data (minimal-change, two modes).

- Unifies the logic of corr_methods.py ("methods" mode) and phase_corr.py ("phase" mode).
- Preserves numerical behavior and file conventions from the originals.
- English-only docstrings and step-wise comments; modularized helpers; main() with argparse.
- Outputs annotated heatmaps and writes permutation logs to text files as before.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.stats import zscore, pearsonr
from nibabel.affines import apply_affine
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components
from brainspace.null_models import MoranRandomization
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["font.family"] = "Arial"


# ----------------------------
# I/O & CIFTI utilities
# ----------------------------

def pull_cifti_data(
    cifti_obj: nib.Cifti2Image
) -> [object, np.ndarray, int]:
    """Load CIFTI image data (float32) and return header, array, and n_time.

    Args:
        cifti_obj: Loaded Cifti2Image.

    Returns:
        (header, data array [time x brain_cols], n_time)
    """
    cifti_obj.set_data_dtype("<f4")
    cifti_data = np.array(cifti_obj.get_fdata())
    cifti_obj.uncache()
    n_time = cifti_data.shape[0]
    return cifti_obj.header, cifti_data, n_time


# ----------------------------
# Data preparation
# ----------------------------

def load_and_slice_maps(
    filepaths: [str], 
    mode: str, 
    comps: int
) -> np.ndarray:
    """Load requested maps from multiple CIFTI files and slice cerebellum (or Grad).

    Behavior mirrors original scripts:
      - "methods" mode includes: CPCA(real maps), Grad1-100, ICA(s/t), HMM, Eigenmap
      - "phase"   mode includes: CPCA(angle maps), Lag, QPP
    The stack order is preserved from the originals.

    Args:
        filepaths: List of CIFTI filepaths in the intended order.
        mode: 'real' or 'phase'.
        comps: Number of CPCA components to keep (top K).

    Returns:
        Stacked 2D array [n_rows x n_features_kept_before_mask].
    """
    collected = []
    for fp in filepaths:
        cifti_obj = nib.load(fp)
        cifti_obj.set_data_dtype("<f4")
        #hdr = cifti_obj.header
        cifti_maps = np.array(cifti_obj.get_fdata())
        cifti_obj.uncache()

        if mode == "methods":
            # CPCA real -> first 'comps' comps from cerebellum
            if "pca" in fp:
                collected.append(cifti_maps[:comps, 65289:83142])
            # Grad (2 maps), different index range
            elif "grad" in fp:
                collected.append(cifti_maps[:2, 70861:88714])
            # ICA (S/T) and HMM (first 3 each) from cerebellum
            elif any(tag in fp for tag in ["ica", "hmm"]):
                collected.append(cifti_maps[:3, 65289:83142])
            # Eigenmap: single map from cerebellum
            elif "eigenmap" in fp:
                collected.append(cifti_maps[0, 65289:83142])
            else:
                raise ValueError(f"Unrecognized file in 'methods' mode: {fp}")

        elif mode == "phase":
            # CPCA angle -> first 'comps' comps from cerebellum
            if "pca" in fp:
                collected.append(cifti_maps[:comps, 65289:83142])
            # Lag & QPP: first 1 map each from cerebellum
            elif any(tag in fp for tag in ["lag", "qpp"]):
                collected.append(cifti_maps[:1, 65289:83142])
            else:
                raise ValueError(f"Unrecognized file in 'phase' mode: {fp}")

        else:
            raise ValueError("mode must be 'methods' or 'phase'")

    return np.vstack(collected)


def build_file_list(
    dataset: str, 
    mode: str
) -> [[str], [str], int]:
    """Construct file list and labels for a given dataset and mode.

    Args:
        dataset: Dataset name string (e.g., 'HCP', 'HCD').
        mode: 'methods' or 'phase'.

    Returns:
        (filepaths, labels_short, comps)
    """
    base_dir = f"./source_files/CPCA_{dataset}/"
    if mode == "methods":
        filepaths = [
            base_dir + f"cpca_cere_{dataset}_3pc_complex_real_results.dtseries.nii",
            base_dir + f"Grad_cere_{dataset}.dtseries.nii",
            base_dir + f"ica_cere_{dataset}_s_results.dtseries.nii",
            base_dir + f"ica_cere_{dataset}_t_results.dtseries.nii",
            base_dir + f"hmm_cere_{dataset}_mean_map_results.dtseries.nii",
            base_dir + f"eigenmap_cere_{dataset}_results.dtseries.nii",
        ]
        labels_short = [
            "Grad 1", "Grad 2",
            "SICA 1", "SICA 2", "SICA 3",
            "TICA 1", "TICA 2", "TICA 3",
            "HMM 1", "HMM 2", "HMM 3",
            "Eignemap",
        ]
        comps = 3

    elif mode == "phase":
        filepaths = [
            base_dir + f"cpca_cere_{dataset}_3pc_complex_ang_results.dtseries.nii",
            base_dir + f"lag_cere_{dataset}_results.dtseries.nii",
            base_dir + f"qpp_cere_{dataset}_results.dtseries.nii",
        ]
        labels_short = [
            "Lag",
            "QPP",
        ]
        comps = 3

    else:
        raise ValueError("mode must be 'methods' or 'phase'")

    return filepaths, labels_short, comps


# ----------------------------
# Moran spectral randomization (nulls)
# ----------------------------

def compute_knn_graph(
    coords: np.ndarray, 
    k: int
):
    """Build a symmetric row-stochastic KNN graph with inverse-distance weights."""
    W_knn = kneighbors_graph(
        coords, n_neighbors=k, mode="distance", include_self=False, metric="euclidean"
    )
    W_knn.data = 1.0 / (W_knn.data + 1e-10)
    W_knn = 0.5 * (W_knn + W_knn.T)
    W_knn = normalize(W_knn, norm="l1", axis=1)
    return W_knn


def moran_null(
    coords: np.ndarray, 
    values: np.ndarray, 
    index: int, 
    n_perm: int, 
    seed: int
) -> np.ndarray:
    """Generate Moran spectral randomizations for a single column x = values[:, index].

    Returns:
        Array of shape (n_perm, n_vertices) with spatially autocorrelated nulls.
    """
    W_knn = compute_knn_graph(coords, k=8)
    n_components, _ = connected_components(W_knn, directed=False)
    if n_components != 1:
        raise RuntimeError("KNN graph is not fully connected; increase k or check coordinates.")

    msr = MoranRandomization(n_rep=n_perm, random_state=seed)
    msr.fit(W_knn)
    x = values[:, index]
    x_null = msr.randomize(x)
    return x_null


def extract_cereb_coords_and_values(
    cifti_fp: str, 
    mni_path: str
) -> [np.ndarray, np.ndarray]:
    """Reproduce the coordinate extraction & value stacking from originals.

    Args:
        cifti_fp: A CIFTI path from which to copy header brain models and values.
        mni_path: Path to an MNI NIfTI template (affine used for ijk -> xyz).

    Returns:
        (coords [n_vox x 3], values [n_vox x time]) as float arrays.
    """
    cifti = nib.load(cifti_fp)
    data = cifti.get_fdata().squeeze()
    bm = cifti.header.get_index_map(1).brain_models

    nifti_template = nib.load(mni_path)
    affine = nifti_template.affine

    coords, values, i_model = [], [], 0
    for model in bm:
        i_model += 1
        # Keep 10th and 11th voxel models, as in the originals
        if model.model_type == "CIFTI_MODEL_TYPE_VOXELS" and (i_model == 10 or i_model == 11):
            ijk = model.voxel_indices_ijk
            xyz = apply_affine(affine, ijk)
            coords.append(xyz)
            start = model.index_offset
            end = start + len(ijk)
            values.append(data[:, start:end])

    coords = np.vstack(coords)
    values = np.concatenate(values, axis=1).T
    return coords, values


# ----------------------------
# Core computation
# ----------------------------

def compute_component_correlations(
    data_stack: np.ndarray,
    comps: int,
    standardize: bool,
) -> pd.DataFrame:
    """Compute |r| between each non-CPCA map and the first 'comps' CPCA components.

    Args:
        data_stack: 2D array stacking CPCA comps first (top 'comps'), followed by other maps.
        comps: Number of CPCA components at the top of the stack.
        standardize: If True, z-score maps across features (as in 'methods' script).

    Returns:
        DataFrame of absolute correlations with columns 'Comp0..Comp{comps-1}'.
    """
    X = data_stack.copy()
    if standardize:
        # Original 'methods' z-scored after zero-mask and then transposed.
        X = zscore(X, axis=1)
    # Compute cross-correlation between target rows and top comps
    # np.corrcoef expects observations in rows -> correlate rows directly
    C = np.corrcoef(X)
    comp_corrs = C[comps:, :comps]
    weights_df = pd.DataFrame(
        comp_corrs,
        columns=[f"Comp{i}" for i in range(comps)],
    )
    return weights_df.abs()


def permutation_significance(
    data_stack_raw: np.ndarray,
    comps: int,
    coords: np.ndarray,
    values: np.ndarray,
    n_perm: int,
    log_prefix: str,
) -> pd.DataFrame:
    """Reproduce the per-cell permutation test via Moran nulls (as in originals).

    Args:
        data_stack_raw: Raw (unstandardized) stacked maps before any z-scoring.
        comps: Number of CPCA components at the top.
        coords: Cerebellar coordinates used to build spatial nulls.
        values: Time x vertices values from the chosen CIFTI for null generation.
        n_perm: Number of nulls.
        log_prefix: File prefix to write observed/null correlations.

    Returns:
        DataFrame of same shape as (n_targets x comps) with strings: '', '*', '**', '***'.
    """
    # Pre-compute nulls for the first three components as in originals
    x_nulls = [moran_null(coords, values, i, n_perm=n_perm) for i in range(3)]

    # Initialize strings with placeholders
    n_targets = data_stack_raw.shape[0] - comps
    sig = pd.DataFrame(
        [["" for _ in range(comps)] for _ in range(n_targets)],
        columns=[f"Comp{i}" for i in range(comps)],
    )

    # Logs
    with open(f"{log_prefix}_r_obs.txt", "w") as f_obs, open(f"{log_prefix}_r_null.txt", "w") as f_null:
        for comp in range(comps):
            for i in range(comps, data_stack_raw.shape[0]):
                r_obs = pearsonr(data_stack_raw[comp], data_stack_raw[i])[0]
                r_null = [pearsonr(x_nulls[comp][n, :], data_stack_raw[i])[0] for n in range(n_perm)]
                p_value = float(np.mean(np.abs(r_null) >= np.abs(r_obs)))

                if p_value < 0.001:
                    star = "***"
                elif p_value < 0.01:
                    star = "**"
                elif p_value < 0.05:
                    star = "*"
                else:
                    star = ""

                sig.iloc[i - comps, comp] = star

    return sig


# ----------------------------
# Plotting
# ----------------------------

def plot_heatmap(
    weights_df_abs_sorted: pd.DataFrame,
    sig_sorted: pd.DataFrame,
    comps: int,
    labels_sorted: [str],
    out_png: str,
    title_bar: str = r"Pearson's $|r|$",
) -> None:
    """Render heatmap with dual-layer text annotations (black under ivory)."""
    fig, ax = plt.subplots(figsize=(2, 5))
    im = ax.imshow(weights_df_abs_sorted.values, aspect="auto", cmap="inferno")
    ax.set_xticks(np.arange(comps))
    ax.set_yticks(np.arange(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.set_xticklabels([f"PC{i+1}" for i in range(comps)], fontsize=8, fontweight="bold")

    # Two-pass text to ensure legibility on colormap
    for i in range(weights_df_abs_sorted.shape[0]):
        for j in range(weights_df_abs_sorted.shape[1]):
            ax.text(
                j + 0.02,
                i + 0.02,
                f"{round(weights_df_abs_sorted.iloc[i, j], 2)}\n{sig_sorted.iloc[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=8,
            )
        for j in range(weights_df_abs_sorted.shape[1]):
            ax.text(
                j,
                i,
                f"{round(weights_df_abs_sorted.iloc[i, j], 2)}\n{sig_sorted.iloc[i, j]}",
                ha="center",
                va="center",
                color="ivory",
                fontweight="bold",
                fontsize=8,
            )

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Adjust position and add colorbar on the right
    box = ax.get_position()
    box.x0 = box.x0 + 0.1
    box.x1 = box.x1 + 0.1
    box.y0 = 0
    box.y1 = 1
    ax.set_position(box)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="15%", pad="25%")
    cax.set_aspect(10)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical", aspect=50)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_title(title_bar, fontsize=8, loc="center", pad=10)

    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------
# Orchestrator
# ----------------------------

def all_correlation(
    dataset: str,
    mode: str,
    output_dir: str = "pic/",
    n_perm: int = 1000,
    mni_path: str = "/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/FSL_MNI152_2mm.nii.gz",
) -> [np.ndarray, [str]]:
    """Compute and plot correlations between CPCA components and other methods.

    Args:
        dataset: Dataset name (e.g., 'HCP', 'HCD').
        mode: 'methods' (Grad/ICA/HMM/Eigenmap) or 'phase' (Lag/QPP vs CPCA angle).
        output_dir: Directory for outputs (PNG and logs).
        n_perm: Number of spatially-constrained permutations (Moran nulls).
        mni_path: Path to MNI template used to derive affine for voxel coordinates.

    Returns:
        (masked_stack_transposed, labels_short)
    """
    # Step 1: Build file list and labels
    filepaths, labels_short, comps = build_file_list(dataset, mode)

    # Step 2: Load maps and apply zero-variance mask
    stack_raw = load_and_slice_maps(filepaths, mode=mode, comps=comps)
    zero_mask = np.std(stack_raw, axis=0) > 0
    stack = stack_raw[:, zero_mask].copy()

    # Step 3: Standardization scheme follows originals
    standardize = True
    # For plotting and correlation matrix, the "methods" script z-scored per row;
    # the "phase" one passed raw (but transposed just before corrcoef). We handle
    # both by compute_component_correlations(standardize=...).

    # Step 4: Coordinates & values for Moran nulls
    coords, values = extract_cereb_coords_and_values(filepaths[0], mni_path)

    # Step 5: Correlations and permutation p-values
    weights_abs = compute_component_correlations(stack, comps=comps, standardize=standardize)
    sig = permutation_significance(stack_raw, comps=comps, coords=coords, values=values, n_perm=n_perm,
                                   log_prefix=f"r_obs{'_phase' if mode=='phase' else '_real'}")

    # Step 6: Sort rows by the column with max |r| (as in originals)
    idx_max = weights_abs.idxmax(axis=1)
    df_list, sig_list, labels_sorted = [], [], []
    for col in weights_abs.columns:
        block_w = weights_abs.loc[idx_max == col]
        block_s = sig.loc[idx_max == col]
        sort_index = block_w[col].sort_values(ascending=False).index
        df_list.append(block_w.loc[sort_index])
        sig_list.append(block_s.loc[sort_index])
        labels_sorted.extend(list(sort_index))
    weights_abs_sorted = pd.concat(df_list, axis=0)
    sig_sorted = pd.concat(sig_list, axis=0)

    # Step 7: Plot and save
    out_png = f"{output_dir}corr_methods_{dataset}{'_phase' if mode=='phase' else '_real'}.png"
    plot_heatmap(
        weights_df_abs_sorted=weights_abs_sorted,
        sig_sorted=sig_sorted,
        comps=comps,
        labels_sorted=labels_sorted,
        out_png=out_png,
        title_bar=r"Pearson's $|r|$",
    )

    # Step 8: Return data for downstream usage (transpose to match prior return)
    # In originals, a transposed array was sometimes returned; we return maps^T
    return stack.T, labels_short


# ----------------------------
# Main
# ----------------------------

def main():
    """CLI entrypoint mirroring the originals with a unified interface."""
    parser = argparse.ArgumentParser(description="Merged correlation analysis (methods/phase).")
    parser.add_argument("--mode", type=str, choices=["real", "phase"], required=True, 
                        help="Analysis mode.")
    parser.add_argument("--dataset", type=str, nargs="+", required=True, 
                        help="Dataset name(s), e.g., HCP HCD.")
    parser.add_argument("--out_dir", type=str, 
                        default="./output/", 
                        help="Output directory for figures and logs.")
    parser.add_argument("--n_perm", type=int, 
                        default=1000, 
                        help="Number of Moran spectral randomizations.")
    parser.add_argument("--mni_path", type=str, 
                        default="./source_files/FSL_MNI152_2mm.nii.gz", 
                        help="Path to MNI template (affine used for voxel ijk->xyz).")
    
    if len(sys.argv) == 1:
        sys.argv += [
            '-mode', 'real',
            '-dataset', 'HCP'
        ]
    args = parser.parse_args()

    all_correlation(
        dataset=args.dataset, 
        mode=args.mode, 
        output_dir=args.out_dir, 
        n_perm=args.n_perm, 
        mni_path=args.mni_path, 
    )


if __name__ == "__main__":
    main()
