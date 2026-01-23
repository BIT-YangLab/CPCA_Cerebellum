import matplotlib.pyplot as plt
import nibabel as nb 
import numpy as np
import os

from glob import glob
from itertools import groupby 
from matplotlib.patches import Rectangle
from nibabel import FileHolder, Cifti2Image, GiftiImage 
from nibabel.gifti.gifti import GiftiDataArray
from scipy.stats import zscore
import nibabel as nib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def assign_gifti_files_to_nested_list(gifti_files):
	id_extract = lambda x: x.split('/')[-1].split('_')[3]
	temp = sorted(gifti_files, key = id_extract) 
	nested_gifti_list = [list(subj_id) for i, subj_id in groupby(temp, id_extract)] 
	return nested_gifti_list

def get_subj_file_list(n_sub, input_type, path, rotate, parcel):
	if input_type == 'cifti':
		subj_files = sorted(glob(f'/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/data/{path}/*.dtseries.nii'))
		#subj_files = sorted(glob(f'/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/data/sleep_300s/filted/{path}*.dtseries.nii'))
	if len(subj_files) < 1:
		raise Exception('No files found in file path')
	if n_sub is None:
		n_sub = len(subj_files)
	if rotate == 0:
		subj_files_sub = subj_files[:n_sub]
	if rotate == 1:
		subj_files_sub = subj_files[:int(n_sub/2)]
		n_sub = int(n_sub/2)
	if rotate == 2:
		subj_files_sub = subj_files[int(n_sub/2):n_sub]
		n_sub = int(n_sub/2)
	return subj_files_sub, n_sub


def load_cifti(cifti_fp):
	cifti = nb.load(cifti_fp)
	return cifti


def load_data_and_stack(n_sub, input_type, path, rotate, parcel=False, verbose=False):
	subj_files, n_sub = get_subj_file_list(n_sub, input_type, path, rotate, parcel)
	subj_files[0], subj_files[1] = subj_files[1], subj_files[0]
	filename = 'group_data.dat'
	subj_file_path = subj_files[0]
	if input_type == 'cifti':
		cifti_obj = load_cifti(subj_file_path)
		_, subj_data, n_time = pull_cifti_data(cifti_obj)
	n_cols = subj_data.shape[1]
	dtype = subj_data.dtype
	del subj_data
	group_data_memmap = np.memmap(filename, dtype=dtype, mode='w+', shape=(n_time * n_sub, n_cols))
	row_indx = 0
	
	for i in tqdm(range(len(subj_files))):
		if verbose:
			print(subj_files[i])
		if input_type == 'cifti':
			subj_file = load_cifti(subj_files[i])
			hdr, subj_data, _ = pull_cifti_data(subj_file)
		group_data_memmap[row_indx:(row_indx+n_time), :] = subj_data
		row_indx += n_time
	del subj_file
	del subj_data
	zero_mask = np.std(group_data_memmap, axis=0) > 0
	zero_mask_indx = np.where(zero_mask)[0]
	group_data_memmap = group_data_memmap[:, zero_mask]
	print('Data loading finished. ')
	return group_data_memmap, hdr, zero_mask_indx, cifti_obj


def pull_cifti_data(cifti_obj):
	cifti_obj.set_data_dtype('<f4')
	cifti_data = np.array(cifti_obj.get_fdata())
	cifti_obj.uncache()
	dtseries_final = cifti_data[:, :64984]
	n_time = dtseries_final.shape[0]
	#n_time = 1800
	return cifti_obj.header, dtseries_final, n_time


def pre_allocate_array(subj_file, input_type, n_sub):
	if input_type == 'cifti':
		subj = load_cifti(subj_file)
	n_rows, n_cols = subj.shape
	group_array = np.empty((n_rows*n_sub, n_cols), np.float32)
	return group_array, n_rows


def write_to_cifti(result, hdr, n_rows, script_name):
	hdr_axis0  = hdr.get_axis(0)
	hdr_axis0.size = n_rows
	hdr_axis1 = hdr.get_axis(1)
	cifti_out = nb.Cifti2Image(result, (hdr_axis0, hdr_axis1))
	nb.save(cifti_out, f'{script_name}_results.dtseries.nii')
