import argparse
import fbpca
import numpy as np
import pickle

from numpy.linalg import pinv
from scipy.signal import hilbert
from scipy.stats import zscore
from utils.utils_full_subjs_srp import load_data_and_stack, write_to_cifti, \
write_to_gifti
import nibabel as nib


def run_main(n_comps, n_sub, indi, path):
    '''
    n_comps: Number of components to be remained
    n_sub: Number of individuals
    base_slices: Spatial slice of cortex and cerebellum
    additional_slices: Spatial slice of added regions
    '''
    analysis_str1 = f'/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_full/srp_full_{indi}'
    analysis_str2 = f'/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_full/crp_full_{indi}'
    phase, hdr, zero_mask = load_data_and_stack(n_sub, indi, path)
    
    tem = nib.load('/home/lvshuo/VDisk4/Lvshuo/visual_module/Yeo7_Yeo-Buckner-Choi-Raut.dlabel.nii').get_fdata()
    
    srp_cere = np.zeros((n_comps, 91282))
    crp_cere = np.zeros((n_comps, 91282))
    for comp in range(n_comps):
        phase_single = phase[comp, :]
        phi_ref = np.median(phase_single[np.where(tem[0]==1)[0]])
        print(len(np.where(tem[0]==1)[0]), phi_ref)
        srp_cere[comp, :] = np.sin(phase_single - phi_ref)
        crp_cere[comp, :] = np.cos(phase_single - phi_ref)
    write_results(srp_cere, crp_cere, n_comps, hdr, zero_mask, analysis_str1, analysis_str2)


def write_results(srp_cere, crp_cere, n_comps, hdr, zero_mask, analysis_str1, analysis_str2):
    voxels = 91282
    cere_s, cere_e = 65289, 83142
    write_to_cifti(srp_cere, hdr, n_comps, f'{analysis_str1}')
    write_to_cifti(crp_cere, hdr, n_comps, f'{analysis_str2}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run main PCA analysis')
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=True,
                        type=int)
    parser.add_argument('-s', '--n_sub',
                        help='Number of subjects to use',
                        default=None,
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
    run_main(args_dict['n_comps'], args_dict['n_sub'], args_dict['individual'], args_dict['path'])
