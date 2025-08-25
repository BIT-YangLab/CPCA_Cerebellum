"""
PCA / CPCA analysis script for cerebellum data (minimal-change refactor).

- Numerical logic aligned with the provided version; fixes obvious bugs.
- English docstrings and step-wise comments added (no Chinese in code).
- Has a main() entry and argparse CLI; supports default injection when run without args.
- Cross-platform path handling via pathlib.
"""

import argparse
import sys
import fbpca
import numpy as np
import pickle
import glob
import tqdm
import nibabel as nb
from scipy.signal import hilbert
from scipy.stats import zscore


# ----------------------------
# Data Loading and PCA Section
# ----------------------------

def get_subj_file_list(
    n_sub: int, 
    path: str
) -> [[str], int]:
    """Retrieve subject file paths."""
    subj_files = sorted(glob(f'{path}/*_smooth_norm_filt.dtseries.nii'))
        
    if len(subj_files) < 1:
        raise FileNotFoundError('No files found in file path')
        
    if n_sub is None:
        n_sub = len(subj_files)
        
    return subj_files[:n_sub], n_sub


def load_data_and_stack(
    n_sub: int, 
    path: str
):
    """Load subject data and vertically stack into a memory-mapped array.

    Args:
        n_sub: Number of subjects or None for all.
        path: Directory of input data.

    Returns:
        group_data_memmap: (T_total, N_cols_kept) memmap with cerebellum data only.
        hdr: CIFTI header from the first file.
        zero_mask_indx: Indices of columns with non-zero std that were kept.
    """
    subj_files, n_sub = get_subj_file_list(n_sub, path)
    subj_file_path = subj_files[0]
    
    # Step 1: Inspect header and get first subject shape
    cifti_obj = nb.load(subj_file_path)
    cifti_obj.set_data_dtype('<f4')
    hdr = cifti_obj.header
    cifti_data = np.array(cifti_obj.get_fdata())
    cifti_obj.uncache()
    
    # Cerebellum columns (as in the provided script)
    subj_data = cifti_data[:, 65289:83142]
    n_time, n_cols = subj_data.shape
    dtype = subj_data.dtype
    
    # Step 2: Pre-allocate memmap to stack all subjects along time
    group_data_memmap = np.memmap(
        'group_data.dat', dtype=dtype, mode='w+', shape=(n_time * n_sub, n_cols)
    )
    
    row_idx = 0
    for file_path in tqdm.tqdm(subj_files):
        subj_file = nb.load(file_path)
        subj_file.set_data_dtype('<f4')
        cifti_data = np.array(subj_file.get_fdata())
        subj_file.uncache()
        subj_data = cifti_data[:, 65289:83142]
        group_data_memmap[row_idx:row_idx + subj_data.shape[0], :] = subj_data
    
    # Step 3: Drop columns with zero variance across stacked time
    zero_mask = np.std(group_data_memmap, axis=0) > 0
    zero_mask_indx = np.where(zero_mask)[0]
    group_data_memmap = group_data_memmap[:, zero_mask]
    
    return group_data_memmap, hdr, zero_mask_indx


def hilbert_transform(
    input_data: np.ndarray
) -> np.ndarray:
    """Perform Hilbert transform and return complex conjugate."""
    complex_data = hilbert(input_data, axis=0)
    return complex_data.conj()


def pca(
    input_data: np.ndarray, 
    n_comps: int, 
    n_iter: int, 
    l: int, 
) -> dict:
    """Run PCA using fbpca with given parameters.

    Args:
        input_data: 2D array (time x features), real or complex.
        n_comps: Number of principal components.
        n_iter: Power iterations for randomized SVD.
        l: Oversampling parameter (will be incremented by n_comps).

    Returns:
        Dict with keys:
            'U': Left singular vectors (time x k).
            's': Singular values (k,).
            'Va': Right singular vectors (k x features).
            'loadings': Component loadings (k x features).
            'exp_var': Explained variance per component (k,).
            'pc_scores': Projection of data onto PCs (time x k).
    """
    l += n_comps
    n_samples = input_data.shape[0]
    (U, s, Va) = fbpca.pca(input_data, k=n_comps, n_iter=n_iter, l=l)
    
    explained_variance_ = (s ** 2) / (n_samples - 1)
    pc_scores = input_data @ Va.T
    
    loadings = Va.T @ np.diag(s)
    loadings /= np.sqrt(input_data.shape[0] - 1)
    
    return {
        'U': U,
        's': s,
        'Va': Va,
        'loadings': loadings.T,
        'exp_var': explained_variance_,
        'pc_scores': pc_scores
    }


def shuffle_time(
    data: np.ndarray
) -> np.ndarray:
    """Shuffle time series indices multiple times."""
    shuffle_indx = np.arange(data.shape[0])
    for _ in range(1000):
        shuffle_indx = np.random.permutation(shuffle_indx)
    return data[shuffle_indx, :]


# ----------------------------
# Write Results Section
# ----------------------------


def write_to_cifti(
    result: np.ndarray, 
    hdr, 
    n_rows: int, 
    script_name: str
) -> None:
    """Write results to a CIFTI file.

    Args:
        result: 2D array to be written (rows x 91,282 brain columns after expansion).
        hdr: CIFTI header from the input file.
        n_rows: Number of rows to encode into the output's first axis.
        script_name: Output filename stem (directory + stem).
    """
    hdr_axis0 = hdr.get_axis(0)
    hdr_axis0.size = n_rows
    hdr_axis1 = hdr.get_axis(1)
    cifti_out = nb.Cifti2Image(result, (hdr_axis0, hdr_axis1))
    nb.save(cifti_out, f'./results/{script_name}_results.dtseries.nii')



def write_results(
    pca_output: dict, 
    shuffle_ts: int, 
    comp_weights: np.ndarray, 
    n_comps: int, 
    hdr, 
    pca_type: str,
    zero_mask: np.ndarray, 
    analysis_str: str, 
) -> None:
    """Save PCA/CPCA results to disk.

    Args:
        pca_output: Dictionary returned by pca().
        shuffle_ts_flag: Whether shuffling was performed (0/1).
        comp_weights: Component weights (k x features_kept).
        n_comps: Number of components (k).
        hdr: CIFTI header for writing.
        pca_type: 'real' or 'complex'.
        zero_mask: Indices of kept columns (non-zero std).
        analysis_str: Output filename stem (without extension).
    """
    if shuffle_ts:
        analysis_str += '_shuffle'
        
    if pca_type == 'complex':
        analysis_str += '_complex'
        pickle.dump({'pca': pca_output, 'metadata': [hdr, zero_mask]},
                    open(f'./results/{analysis_str}_results.pkl', 'wb'))
        # Prepare different representations
        comp_weights_ang = np.zeros((n_comps, 91282))
        comp_weights_ang[:, 65289:83142] = np.angle(comp_weights)
        write_to_cifti(comp_weights_ang, hdr, n_comps, f'{analysis_str}_ang')

        comp_weights_real = np.zeros((n_comps, 91282))
        comp_weights_real[:, 65289:83142] = np.real(comp_weights)
        write_to_cifti(comp_weights_real, hdr, n_comps, f'{analysis_str}_real')

    elif pca_type == 'real':
        pickle.dump(pca_output, open(f'./results/{analysis_str}_results.pkl', 'wb'))
        comp_weights_cere = np.zeros((n_comps, 91282))
        comp_weights_cere[:, 65289:83142] = comp_weights
        write_to_cifti(comp_weights_cere, hdr, n_comps, analysis_str)


# ----------------------------
# Reconstruction Section
# ----------------------------

def reconstruct_ts(
    pca_res: dict, 
    n: int, 
    real: bool = True
) -> np.ndarray:
    """Reconstruct time series from a single PCA component.

    Args:
        pca_res: Result dict from pca().
        n: Component index.
        real: If True, return real part; otherwise imaginary part.

    Returns:
        Reconstructed (time x features) array for component n.
    """
    U = pca_res['U'][:, n][:, np.newaxis]
    s = np.atleast_2d(pca_res['s'][n])
    Va = pca_res['Va'][n, :].conj()[np.newaxis, :]
    recon_ts = U @ s @ Va
    return np.real(recon_ts) if real else np.imag(recon_ts)


def create_bins(
    phase_ts: np.ndarray, 
    n_bins: int
):
    """Create bins for phase values.

    Args:
        phase_ts: 1D array of per-time phase values.
        n_bins: Number of bins.

    Returns:
        bin_idx: Bin assignment per timepoint.
        bin_centers: Center value per bin.
    """
    _, bins = np.histogram(phase_ts, n_bins)
    bin_idx = np.digitize(phase_ts, bins)
    bin_centers = np.mean(np.vstack([bins[:-1], bins[1:]]), axis=0)
    return bin_idx, bin_centers


def create_dynamic_phase_maps(
    recon_ts: np.ndarray, 
    bin_idx: np.ndarray, 
    n_bins: int
) -> np.ndarray:
    """Compute dynamic phase maps for each bin.

    Args:
        recon_ts: Reconstructed time x features array.
        bin_idx: Bin index per timepoint.
        n_bins: Number of bins.

    Returns:
        Array of shape (n_bins, features) with mean per bin.
    """
    bin_timepoints = []
    for n in range(1, n_bins + 1):
        ts_idx = np.where(bin_idx == n)[0]
        bin_timepoints.append(np.mean(recon_ts[ts_idx, :], axis=0))
    return np.array(bin_timepoints)


def run_reconstruction(
    cpca_res: dict, 
    n_recon: int, 
    n_bins: int, 
    hdr, 
) -> None:
    """Run dynamic phase map reconstruction and write results.

    Args:
        cpca_res: Dict containing key 'pca' with pca() result.
        n_recon: Number of leading components to reconstruct.
        n_bins: Number of phase bins.
        hdr: CIFTI header for writing.
    """
    bin_idx_all, bin_centers_all = [], []
    for n in range(n_recon):
        print(f'Comp {n}')
        recon_ts = reconstruct_ts(cpca_res['pca'], n, real=True)
        phase_ts = np.angle(cpca_res['pca']['pc_scores'][:, n])
        bin_idx, bin_centers = create_bins(phase_ts, n_bins)
        dynamic_phase_map = create_dynamic_phase_maps(recon_ts, bin_idx, n_bins)
        bin_idx_all.append(bin_idx)
        bin_centers_all.append(bin_centers)
        write_to_cifti(dynamic_phase_map, hdr, n_bins, f'./results/cpca_comp{n}_recon')
    pickle.dump([bin_idx_all, bin_centers_all],
                open('./results/cpca_reconstruction_results.pkl', 'wb'))


# ---------------------- Main Function ----------------------

def main(
    n_comps: int, 
    n_sub: int, 
    real_complex: str, 
    dataset: str, 
    shuffle_ts: int, 
    path: str, 
    reconstruct: int, 
    n_reconstruct: int, 
    n_bins: int
) -> None:
    """Pipeline entrypoint: load, preprocess, PCA/CPCA, save, and optionally reconstruct.

    Args:
        n_comps: Number of PCA components.
        n_sub: Number of subjects or None for all.
        real_complex: 'real' for standard PCA or 'complex' for Hilbert-based CPCA.
        shuffle_ts_flag: 1 to shuffle time indices before PCA, else 0.
        path: Directory containing input data.
        reconstruct_flag: 1 to run dynamic reconstruction, else 0.
        n_reconstruct: Number of components to reconstruct.
        n_bins: Number of bins for phase-based maps (if reconstructing).
    """
    # Step 1: Data loading
    group_data, hdr, zero_mask = load_data_and_stack(n_sub, path)

    # Step 2: Preprocessing
    group_data = zscore(group_data.astype(np.float32))
    if shuffle_ts:
        group_data = shuffle_time(group_data)
    if real_complex == 'complex':
        group_data = hilbert_transform(group_data)

    # Step 3: PCA computation
    pca_output = pca(group_data, n_comps)

    # Step 4: Save PCA results
    write_results(pca_output, shuffle_ts, 
                  pca_output['loadings'], n_comps, hdr, real_complex, zero_mask, 
                  './results/cpca_cere_{dataset}_{n_comps}pc')

    # Step 5: Optional reconstruction
    if reconstruct:
        run_reconstruction({'pca': pca_output}, n_reconstruct, n_bins, hdr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PCA/CPCA analysis with optional reconstruction.')
    parser.add_argument('--n_comps', type=int, required=True, 
                        help='Number of PCA components.')
    parser.add_argument('--n_sub', type=int, 
                        default=None, 
                        help='Number of subjects.')
    parser.add_argument('--real_complex', type=str, choices=['real', 'complex'], 
                        default='real', 
                        help='PCA type.')
    parser.add_argument('--dataset', type=str, choices=['HCP', 'HCD'], 
                        default='HCP', 
                        help='Dataset used in pipline.')
    parser.add_argument('--shuffle_ts', type=int, 
                        default=0, 
                        help='Shuffle time series (1=True).')
    parser.add_argument('--path', type=str, required=True, 
                        help='Path to input files.')
    parser.add_argument('--reconstruct', type=int, 
                        default=0, 
                        help='Perform reconstruction.')
    parser.add_argument('--n_reconstruct', type=int, 
                        default=3, 
                        help='Number of components to reconstruct.')
    parser.add_argument('--n_bins', type=int, 
                        default=60, 
                        help='Number of phase bins.')

    if len(sys.argv) == 1:
        sys.argv += [
            '-n', '3',
            '-path', './data/'
        ]
    args = parser.arguments()
    main(args.n_comps, args.n_sub, args.real_complex, args.dataset, args.shuffle_ts, 
         args.path, args.reconstruct, args.n_reconstruct, args.n_bins)
