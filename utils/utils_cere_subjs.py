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

def get_subj_file_list(n_sub, indi, path):
	subj_files = sorted(glob(f'/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/data/{path}/{indi}*.nii'))
	print(subj_files)
	if len(subj_files) < 1:
		raise Exception('No files found in file path')
	if n_sub is None:
		n_sub = len(subj_files)
	subj_files_sub = subj_files[:n_sub]
	return subj_files_sub, n_sub


def load_cifti(cifti_fp):
	cifti = nb.load(cifti_fp)
	return cifti


def load_gifti(gifti_fps):
	gifti_LR = []
	file_base = gifti_fps[0].split('.')[0]
	for hem in ['L', 'R']:
		indx = gifti_fps.index(f'{file_base}.{hem}.func.gii')
		gifti_LR.append(nb.load(gifti_fps[indx]))
	return gifti_LR


def load_data_and_stack(n_sub, indi, path):
	subj_files, n_sub = get_subj_file_list(n_sub, indi, path)
	filename = 'group_data.dat'
	subj_file_path = subj_files[0]
	cifti_obj = load_cifti(subj_file_path)
	_, subj_data, n_time = pull_cifti_data(cifti_obj)
	n_cols = subj_data.shape[1]
	dtype = subj_data.dtype
	del subj_data
	group_data_memmap = np.memmap(filename, dtype=dtype, mode='w+', shape=(n_time * n_sub, n_cols))
	row_indx = 0
	
	for i in tqdm(range(len(subj_files))):
		subj_file = load_cifti(subj_files[i])
		hdr, subj_data, _ = pull_cifti_data(subj_file)
		group_data_memmap[row_indx:(row_indx+n_time), :] = subj_data
		row_indx += n_time
	
	zero_mask = np.std(group_data_memmap, axis=0) > 0
	zero_mask_indx = np.where(zero_mask)[0]
	#group_data_memmap = group_data_memmap[:, zero_mask]
	#group_data_memmap = np.nan_to_num(group_data_memmap, nan=0)
	has_nan = np.isnan(group_data_memmap).any()
	print('Data loading finished. ')
	return group_data_memmap, hdr, zero_mask_indx


def pull_cifti_data(cifti_obj):
	cifti_obj.set_data_dtype('<f4')
	cifti_data = np.array(cifti_obj.get_fdata())
	cifti_obj.uncache()

	dtseries_final = cifti_data[:, 65289:83142]
	#dtseries_final = cifti_data[:, 21236:23480]
	n_time = dtseries_final.shape[0]
	return cifti_obj.header, dtseries_final, n_time


def pull_gifti_data(giftis):
	# LH and RH data are concatenated with LH first and RH second!
	gifti_L_array = np.array(giftis[0].agg_data())
	gifti_R_array = np.array(giftis[1].agg_data())
	gifti_all = np.concatenate((gifti_L_array, gifti_R_array), axis=1)
	n_time = gifti_all.shape[0]
	n_vert_L = gifti_L_array.shape[1]
	n_vert_R = gifti_R_array.shape[1]
	gifti_all = gifti_all[::]
	return gifti_all, n_time, n_vert_L, n_vert_R


def pre_allocate_array(subj_file, n_sub):
	subj = load_cifti(subj_file)	
	n_rows, n_cols = subj.shape
	group_array = np.empty((n_rows*n_sub, n_cols), np.float64)
	return group_array, n_rows


def write_to_cifti(result, hdr, n_rows, script_name):
	hdr_axis0  = hdr.get_axis(0)
	hdr_axis0.size = n_rows
	hdr_axis1 = hdr.get_axis(1)
	cifti_out = nb.Cifti2Image(result, (hdr_axis0, hdr_axis1))
	nb.save(cifti_out, f'{script_name}_results.dtseries.nii')


def write_to_gifti(result, giftis, script_name, zero_mask, cifti=True):
	example_array_L = giftis[0].darrays[0]
	example_array_R = giftis[1].darrays[0]		
	L_shape = len(giftis[0].agg_data()[0])
	padded_result = np.zeros([result.shape[0], L_shape*2])
	padded_result[:, zero_mask] = result
	L_result = padded_result[:, :L_shape]
	R_result = padded_result[:, L_shape:]
	L_gifti_image = GiftiImage(meta=giftis[0].meta)
	R_gifti_image = GiftiImage(meta=giftis[1].meta)
	for row_L, row_R in zip(L_result, R_result):
		gifti_array_L = GiftiDataArray(row_L, intent=example_array_L.intent,
			datatype=example_array_L.datatype, meta=example_array_L.meta)
		gifti_array_R = GiftiDataArray(row_R, intent=example_array_R.intent,
			datatype=example_array_R.datatype, meta=example_array_R.meta)
		L_gifti_image.add_gifti_data_array(gifti_array_L)
		R_gifti_image.add_gifti_data_array(gifti_array_R)
	nb.save(L_gifti_image, f'{script_name}.L.func.gii')
	nb.save(R_gifti_image, f'{script_name}.R.func.gii')
	if cifti:
		os.system(f'./algo/utils/giftis_to_cifti.sh {script_name}.L.func.gii '
		f'{script_name}.R.func.gii {script_name}.dtseries.nii')
