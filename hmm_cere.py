import argparse
import numpy as np
import pickle
import fbpca
import nibabel as nib

from utils.utils_cere import load_data_and_stack, write_to_cifti
from scipy.stats import zscore
from hmmlearn import hmm


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


def run_main(n_comps, n_sub, input_type, cov_type, path):
    analysis_str = f'results_cere/hmm_cere_{path}'
    group_data, hdr, zero_mask = load_data_and_stack(n_sub, input_type, path)
    group_data = group_data.astype(np.float32)
    group_data = zscore(group_data)
    pca_output = pca(group_data, 3)
    hmm_model, state_ts, mean_maps = gmm_hmm(pca_output, n_comps, cov_type) 
    write_results(input_type, [hmm_model, state_ts], mean_maps, analysis_str, n_comps, hdr, zero_mask)


def gmm_hmm(pca_output, n_comps, cov_type):
    ghmm = hmm.GMMHMM(n_components=n_comps, covariance_type=cov_type, n_iter=200)
    scores = pca_output['pc_scores']
    ghmm.fit(scores)
    pred_labels = ghmm.predict(scores)
    # Project hidden state mean vectors to original dimensions
    mean_maps = np.squeeze(ghmm.means_) @ pca_output['Va']
    return ghmm, pred_labels, mean_maps


def write_results(input_type, hmm_results, mean_maps, analysis_str, n_comps, hdr, zero_mask):
    pickle.dump(hmm_results, 
                open(f'{analysis_str}_results.pkl', 'wb'))
    if n_comps==1:
        mean_maps = mean_maps[np.newaxis, :]

    comp_weights = np.zeros((n_comps, 91282))
    comp_weights[:, 65289:83142] = mean_maps

    if input_type == 'cifti':
        write_to_cifti(comp_weights, hdr, n_comps, f'{analysis_str}_mean_map')
    elif input_type == 'gifti':
        write_to_gifti(comp_weights, hdr, f'{analysis_str}_mean_map', zero_mask)


if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Estimate Gaussian Mixture HMM')
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components for HMM',
                        required=True,
                        type=int)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
                        type=int)
    parser.add_argument('-i', '--input_type',
                        help='Whether to load resampled metric .gii files or '
                        'full cifti files',
                        choices=['cifti', 'gifti'],
                        required=False,
                        default='cifti',
                        type=str)
    parser.add_argument('-c', '--cov_type',
                        help='Size of window for time delay embedding',
                        default='full',
                        choices=['diag', 'full', 'tied'],
                        required=False,
                        type=str)
    parser.add_argument('-path', '--path',
                        help='Whether',
                        default='',
                        required=True,
                        type=str)
    
    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['n_sub'], args_dict['input_type'], args_dict['cov_type'], args_dict['path'])
