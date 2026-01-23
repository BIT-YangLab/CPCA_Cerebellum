import argparse
import fbpca
import numpy as np
import pickle

from numpy.linalg import pinv
from scipy.signal import hilbert
from scipy.stats import zscore
from utils.utils_full_subjs import load_data_and_stack, write_to_cifti, \
write_to_gifti
from utils.rotation import varimax, promax
import nibabel as nib


def hilbert_transform(input_data):
    complex_data = hilbert(input_data, axis=0)
    return complex_data.conj()


def pca(input_data, n_comps, n_iter=100, l=10):
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


def run_main(n_comps, n_sub, global_signal, rotate, 
            input_type, pca_type, center, shuffle_ts, 
            simulate_var, indi, path):
    '''
    n_comps: Number of components to be remained
    n_sub: Number of individuals
    base_slices: Spatial slice of cortex and cerebellum
    additional_slices: Spatial slice of added regions
    '''
    analysis_str = f'results_full/cpca_full_{indi}'
    group_data, hdr, zero_mask = load_data_and_stack(n_sub, indi, path)
    group_data = zscore(group_data)
    if shuffle_ts:
        group_data = shuffle_time(group_data)
    if simulate_var:
        group_data = var_simulate(group_data, group_data.shape[0])
    if center == 'r':
        group_data -= group_data.mean(axis=1, keepdims=True)
    if pca_type == 'complex':
        group_data = hilbert_transform(group_data)
    group_data = np.nan_to_num(group_data, nan=0)
    has_nan = np.isnan(group_data).any()
    print(has_nan)
    pca_output = pca(group_data, n_comps)
    if rotate is not None:
        pca_output = rotation(pca_output, group_data, rotate)

    write_results(input_type, pca_output, rotate, shuffle_ts, simulate_var, pca_output['loadings'], n_comps, 
                hdr, pca_type, global_signal, zero_mask, analysis_str, '')
    del pca_output
    del group_data
    

def rotation(pca_output, group_data, rotation):
    if rotation == 'varimax':
        rotated_weights, _ = varimax(pca_output['loadings'].T)
    elif rotation == 'promax':
        rotated_weights, _, _ = promax(pca_output['loadings'].T)
    projected_scores = group_data @ pinv(rotated_weights).T
    pca_output['loadings'] = rotated_weights.T
    pca_output['pc_scores'] = projected_scores
    return pca_output


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
                n_comps, hdr, pca_type, global_signal, zero_mask, analysis_str, sli):
    if global_signal:
        analysis_str += '_gs'
    if rotate:
        analysis_str += f'_{rotate}'
    if shuffle_ts:
        analysis_str += '_shuffle'
    if simulate_var:
        analysis_str += '_VAR(1)'
    if pca_type == 'complex':
        pickle.dump({
                    'pca': pca_output, 
                    'metadata': [input_type, hdr, zero_mask]
                    }, open(f'{analysis_str}_results.pkl', 'wb'))
        voxels = 91282
        #voxels = 24497
        cere_s, cere_e = 65289, 83142
        #cere_s, cere_e = 21236, 23480
        comp_weights_amp_cere = np.abs(comp_weights)
        comp_weights_ang_cere = np.angle(comp_weights)
        comp_weights_real_cere = np.real(comp_weights)
        comp_weights_imag_cere = np.imag(comp_weights)
        if input_type == 'cifti':
            write_to_cifti(comp_weights_ang_cere, hdr, n_comps, f'{analysis_str}_ang')
            write_to_cifti(comp_weights_amp_cere, hdr, n_comps, f'{analysis_str}_amp')
            write_to_cifti(comp_weights_real_cere, hdr, n_comps, f'{analysis_str}_real')
            write_to_cifti(comp_weights_imag_cere, hdr, n_comps, f'{analysis_str}_imag')
    elif pca_type == 'real':
        pickle.dump(pca_output, open(f'{analysis_str}_results.pkl', 'wb'))
        if input_type == 'cifti':
            write_to_cifti(comp_weights, hdr, n_comps, analysis_str)


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
    parser.add_argument('-g', '--gs_regress',
                        help='Whether to use global signal regressed data',
                        default=0,
                        required=False,
                        type=bool)
    parser.add_argument('-r', '--rotate',
                        help='Whether to rotate pca weights',
                        default=None,
                        required=False,
                        choices=['varimax', 'promax'],
                        type=str)
    parser.add_argument('-i', '--input_type',
                        help='Whether to load resampled metric .gii files or '
                        'full cifti files',
                        choices=['cifti', 'gifti'],
                        required=False,
                        default='cifti',
                        type=str)
    parser.add_argument('-p', '--real_complex',
                        help='Calculate complex or real PCA',
                        default='real',
                        choices=['real', 'complex'],
                        type=str)
    parser.add_argument('-c', '--center',
                        help='Whether to center along the columns (c) or rows (r)',
                        default='c',
                        choices=['c','r'],
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
    parser.add_argument('-indiname', '--individual', 
                        default=0,
                        required=False,
                        type=str)
    parser.add_argument('-path', '--path',
                        help='Whether',
                        default='',
                        required=True,
                        type=str)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['n_sub'], 
            args_dict['gs_regress'], args_dict['rotate'], 
            args_dict['input_type'], args_dict['real_complex'], 
            args_dict['center'], args_dict['shuffle_ts'], 
            args_dict['simulate_var'], args_dict['individual'], 
            args_dict['path'])
