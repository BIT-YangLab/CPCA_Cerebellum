import argparse
import fbpca
import numpy as np
import pickle

from numpy.linalg import pinv
from scipy.signal import hilbert
from scipy.stats import zscore
from utils.utils_full import load_data_and_stack, write_to_cifti
from utils.rotation import varimax, promax
import nibabel as nib


def save_as_memmap(arr, filename, dtype=np.complex64):
    fp = np.memmap(filename, dtype=dtype, mode='w+', shape=arr.shape)
    fp[:] = arr[:]
    del fp

def hilbert_transform(input_data):
    complex_data = hilbert(input_data.astype(np.float32), axis=0)
    return complex_data.astype(np.complex64).conj()
    
def hilbert_block(data, block=2000):
    nT, nV = data.shape
    out = np.zeros((nT, nV), dtype=np.complex64)
    for i in range(0, nV, block):
        if i+block > 91282:
            block_data = data[:, i:]
            out[:, i:] = hilbert(block_data, axis=0).astype(np.complex64)
        else:
            block_data = data[:, i:i+block]
            out[:, i:i+block] = hilbert(block_data, axis=0).astype(np.complex64)
    return out

def pca(input_data, n_comps, n_iter=100, l=5):
    l += n_comps
    n_samples = input_data.shape[0]
    (U, s, Va) = fbpca.pca(input_data, k=n_comps, n_iter=n_iter, l = l)
    explained_variance_ = (s ** 2) / (n_samples - 1)
    pc_scores = input_data @ Va.T
    loadings =  Va.T @ np.diag(s) 
    loadings /= np.sqrt(input_data.shape[0]-1)
    output_dict = {
        'U': U,
        's': s,
        'Va': Va,
        'loadings': loadings.T,
        'exp_var': explained_variance_,
        'pc_scores': pc_scores
        }   
    return output_dict


def run_main(n_comps, n_sub, rotate, 
            input_type, pca_type, shuffle_ts,
            simulate_var, path):
    '''
    n_comps: Number of components to be remained
    n_sub: Number of individuals
    '''
    analysis_str = f'/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_full/cpca_full_{n_comps}pcs_{path}'
    print(analysis_str)
    group_data, hdr, zero_mask, cifti_obj = load_data_and_stack(n_sub, input_type, path, rotate)
    group_data = group_data.astype(np.float32)
    group_data = zscore(group_data).astype(np.float32)
    print(group_data.shape)
    if shuffle_ts:
        group_data = shuffle_time(group_data)
    if simulate_var:
        group_data = var_simulate(group_data, group_data.shape[0])
    if pca_type == 'complex':
        #group_data = hilbert_transform(group_data)
        group_data = hilbert_block(group_data)
    print('HT done. ')
    save_as_memmap(group_data, "group_data_hilbert.dat")
    del group_data
    A = np.memmap("group_data_hilbert.dat", dtype=np.complex64, mode='r', shape=(int(2400*50), 82837))
    pca_output = pca(A, n_comps)

    write_results(input_type, pca_output, rotate, shuffle_ts, simulate_var, pca_output['loadings'], n_comps, hdr, pca_type, zero_mask, analysis_str, cifti_obj)
    del pca_output


def shuffle_time(data):
    shuffle_indx = np.arange(data.shape[0])
    for i in range(100):
        shuffle_indx = np.random.permutation(shuffle_indx)
    return data[shuffle_indx, :]


def var_simulate(data, n_simulate, pca_n=200):
    pca_dim_res = pca(data, pca_n)
    var = VAR(pca_dim_res['pc_scores'])
    var_res = var.fit(maxlags=1)
    data_sim = var_res.simulate_var(n_simulate)
    data_sim = data_sim @ pca_dim_res['Va']
    return data_sim



def write_results(input_type, pca_output, rotate, shuffle_ts, simulate_var, comp_weights, 
                n_comps, hdr, pca_type, zero_mask, analysis_str, cifti_obj):
    if cifti_obj.shape[1] == 91282:
        cere_start, cere_end, length_full = 65289, 83142, 91282
    else:
        cere_start, cere_end, length_full = 70861, 88714, 96854
    if shuffle_ts:
        analysis_str += '_shuffle'
    if rotate:
        analysis_str += f'_{rotate}'
    if pca_type == 'complex':
        analysis_str += '_complex'
        pickle.dump({
                    'pca': pca_output, 
                    'metadata': [input_type, hdr, zero_mask]
                    }, open(f'{analysis_str}_results.pkl', 'wb'))
        comp_weights_amp_cere = np.abs(comp_weights)
        comp_weights_amp = np.zeros((n_comps, length_full))
        comp_weights_amp[:, :64984] = comp_weights_amp_cere[:, :64984]
        comp_weights_amp[:, 65289:83142] = comp_weights_amp_cere[:, 64984:]
        comp_weights_ang_cere = np.angle(comp_weights)
        comp_weights_ang = np.zeros((n_comps, length_full))
        comp_weights_ang[:, :64984] = comp_weights_ang_cere[:, :64984]
        comp_weights_ang[:, 65289:83142] = comp_weights_ang_cere[:, 64984:]
        comp_weights_real_cere = np.real(comp_weights)
        comp_weights_real = np.zeros((n_comps, length_full))
        comp_weights_real[:, :64984] = comp_weights_real_cere[:, :64984]
        comp_weights_real[:, 65289:83142] = comp_weights_real_cere[:, 64984:]
        comp_weights_imag_cere = np.imag(comp_weights)
        comp_weights_imag = np.zeros((n_comps, length_full))
        comp_weights_imag[:, :64984] = comp_weights_imag_cere[:, :64984]
        comp_weights_imag[:, 65289:83142] = comp_weights_imag_cere[:, 64984:]
        if input_type == 'cifti' or input_type == 'indice':
            write_to_cifti(comp_weights_ang, hdr, n_comps, f'{analysis_str}_ang')
            write_to_cifti(comp_weights_amp, hdr, n_comps, f'{analysis_str}_amp')
            write_to_cifti(comp_weights_real, hdr, n_comps, f'{analysis_str}_real')
            write_to_cifti(comp_weights_imag, hdr, n_comps, f'{analysis_str}_imag')
    elif pca_type == 'real':
        pickle.dump(pca_output, open(f'{analysis_str}_results.pkl', 'wb'))
        comp_weights_cere = np.zeros((n_comps, length_full))
        comp_weights_cere[:, cortex_start:cortex_end] = comp_weights
        if input_type == 'cifti' or input_type == 'indice':
            write_to_cifti(comp_weights_cere, hdr, n_comps, analysis_str)


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run main PCA analysis')
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=True,
                        type=int)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    parser.add_argument('-r', '--rotate',
                        help='Whether to rotate pca weights',
                        default=0,
                        required=False,
                        type=int)
    parser.add_argument('-i', '--input_type',
                        help='Whether to load resampled metric .gii files or '
                        'full cifti files',
                        choices=['cifti', 'gifti', 'indice'],
                        required=False,
                        default='cifti',
                        type=str)
    parser.add_argument('-p', '--real_complex',
                        help='Calculate complex or real PCA',
                        default='real',
                        choices=['real', 'complex'],
                        type=str)
    parser.add_argument('-null_shuffle', '--shuffle_ts',
                        help='Whether to shuffle time series (preserving '
                            'correlation structure but removing lag)',
                        default=0,
                        required=False,
                        type=int)
    parser.add_argument('-null_var', '--simulate_var',
                        help='Whether to to perform on simulated VAR(1) fit on data '
                        ' (preserving both correlation structure and (some) lag)',
                        default=0,
                        required=False,
                        type=int)
    parser.add_argument('-path', '--path',
                        help='Whether',
                        default='',
                        required=True,
                        type=str)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['n_sub'], 
            args_dict['rotate'], 
            args_dict['input_type'], args_dict['real_complex'], 
            args_dict['shuffle_ts'], 
            args_dict['simulate_var'], args_dict['path'])
