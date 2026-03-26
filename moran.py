import numpy as np
import nibabel as nib
from brainspace.null_models import MoranRandomization
from nibabel.affines import apply_affine
from sklearn.neighbors import kneighbors_graph
from brainspace.null_models import MoranRandomization
from scipy.sparse.csgraph import connected_components
from brainspace.null_models.moran import compute_mem
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
from scipy.stats import pearsonr

def pull_cifti_data(cifti_obj):
    cifti_obj.set_data_dtype('<f4')
    cifti_data = np.array(cifti_obj.get_fdata())
    n_time = cifti_data.shape[0]
    return cifti_obj.header, cifti_data, n_time

file_dir = '/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_cere/CPCA_HCP_100/'
cifti_fps = (
    file_dir+'cpca_cere_10pcs_HCP_complex_real_results.dtseries.nii', 
    file_dir+'Grad1-100.dtseries.nii', 
    file_dir+'ica_cere_HCP_s_results.dtseries.nii', 
    file_dir+'ica_cere_HCP_t_results.dtseries.nii', 
    file_dir+'hmm_cere_HCP_mean_map_results.dtseries.nii', 
    file_dir+'eigenmap_cere_HCP_results.dtseries.nii'
    )






# ¼ÓÔØÊý¾Ý
cifti = nib.load('/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_cere/CPCA_HCP_100/cpca_cere_10pcs_HCP_complex_real_results.dtseries.nii')
data = cifti.get_fdata().squeeze()
bm = cifti.header.get_index_map(1).brain_models
nifti_template = nib.load("/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/data/FSL_MNI152_2mm.nii.gz")
affine = nifti_template.affine

# ÌáÈ¡ subcortical ²¿·Ö×ø±êºÍÖµ
coords = []
values = []

i = 0
for model in bm:
    i += 1
    if model.model_type == "CIFTI_MODEL_TYPE_VOXELS" and (i==10 or i==11):
        print(i)
        ijk = model.voxel_indices_ijk
        xyz = apply_affine(affine, ijk)
        coords.append(xyz)
        start = model.index_offset
        end = start + len(ijk)
        print(start, end, end-start)
        values.append(data[:, start:end])

coords = np.vstack(coords)
values = np.concatenate(values, axis=1).T

# ·½·¨1£ºÊ¹ÓÃKNNÍ¼¹¹½¨È¨ÖØ¾ØÕó£¨ÍÆ¼ö£©
print("·½·¨1£ºÊ¹ÓÃKNNÍ¼")
k = 7
# Ê¹ÓÃsklearnµÄkneighbors_graph£¬Ëü»á×Ô¶¯´¦ÀíÁ¬Í¨ÐÔ
W_knn = kneighbors_graph(coords, n_neighbors=k, mode='distance', 
                         include_self=False, metric='euclidean')

# ×ª»»ÎªÈ¨ÖØ£¨¾àÀëµÄµ¹Êý£©
W_knn.data = 1.0 / (W_knn.data + 1e-10)  # ±ÜÃâ³ýÁã

# ¶Ô³Æ»¯
W_knn = 0.5 * (W_knn + W_knn.T)

# ÐÐ¹éÒ»»¯
W_knn = normalize(W_knn, norm='l1', axis=1)

# ¼ì²éÁ¬Í¨ÐÔ
n_components, labels = connected_components(W_knn, directed=False)

n_perm = 1000

if n_components == 1:
    try:
        msr = MoranRandomization(n_rep=n_perm, random_state=42)
        msr.fit(W_knn)
        x = values[:, 1]
        x_null = msr.randomize(x)
        print(f"Shape of x_null: {x_null.shape}")
        
    except ValueError as e:
        print(f"KNN·½·¨Ê§°Ü: {e}")
        print("³¢ÊÔ·½·¨2...")




for comp in range(3):
    for i in range(1, len(cifti_fps)):
        if i==5:
            c = 1
        else:
            c = 3
        for j in range(c):
            hdr, cifti_map, n_time = pull_cifti_data(nib.load(cifti_fps[i]))
            if i==1:
                cifti_map = cifti_map[j, 70861:88714]
            else:
                cifti_map = cifti_map[j, 65289:83142]
            r_obs = pearsonr(values[:, comp], cifti_map)[0]
            r_null = [pearsonr(x_null[n, :], cifti_map)[0] for n in range(n_perm)]
            p_value = np.mean(np.abs(r_null) >= np.abs(r_obs))
            print(f"PC{comp+1}  Pattern{i}  Comp{j+1}", p_value)




'''
# ·½·¨2£ºÐÞÕýÔ­Ê¼¾àÀëÈ¨ÖØ¾ØÕó
print("\n·½·¨2£ºÐÞÕý¾àÀëÈ¨ÖØ¾ØÕó")
D = squareform(pdist(coords))

# Ê¹ÓÃ¸ü±£ÊØµÄkÖµºÍ²»Í¬µÄÈ¨ÖØ¹¹½¨·½Ê½
k = 6  # ¼õÉÙÁÚ¾ÓÊý
W_dist = np.zeros_like(D)

for i in range(W_dist.shape[0]):
    # ÕÒµ½k¸ö×î½üÁÚ£¨ÅÅ³ý×Ô¼º£©
    distances = D[i]
    neighbors = np.argsort(distances)[1:k+1]  # ÅÅ³ý¾àÀëÎª0µÄ×Ô¼º
    
    # Ê¹ÓÃ¸ßË¹ºËÈ¨ÖØ¶ø²»ÊÇ¼òµ¥µÄµ¹Êý
    sigma = np.median(distances[neighbors])  # ×ÔÊÊÓ¦bandwidth
    W_dist[i, neighbors] = np.exp(-distances[neighbors]**2 / (2 * sigma**2))

# ¶Ô³Æ»¯
W_dist = 0.5 * (W_dist + W_dist.T)

# ÐÐ¹éÒ»»¯
row_sums = W_dist.sum(axis=1)
row_sums[row_sums == 0] = 1  # ±ÜÃâ³ýÁã
W_dist = W_dist / row_sums[:, np.newaxis]

# ×ª»»ÎªÏ¡Êè¾ØÕó
W_dist_sparse = sparse.csr_matrix(W_dist)

print(f"¾àÀëÈ¨ÖØ¾ØÕóÐÎ×´: {W_dist_sparse.shape}")
print(f"¾àÀëÈ¨ÖØ¾ØÕó·ÇÁãÔªËØÊý: {W_dist_sparse.nnz}")

# ¼ì²éÁ¬Í¨ÐÔ
n_components, labels = connected_components(W_dist_sparse, directed=False)
print(f"Á¬Í¨·ÖÁ¿Êý: {n_components}")

if n_components == 1:
    try:
        mem_dist, mev_dist = compute_mem(W_dist_sparse)
        print(f"³É¹¦¼ÆËãMEM£¬ÌØÕ÷Öµ·¶Î§: [{np.min(mev_dist):.6f}, {np.max(mev_dist):.6f}]")
        print(f"MEMÐÎ×´: {mem_dist.shape}")
        
        # ½øÐÐMoranËæ»ú»¯
        msr2 = MoranRandomization(n_rep=1, random_state=42)
        msr2.fit(W_dist_sparse)
        x = values[:, 0]
        x_null2 = msr2.randomize(x)
        print(f"Ëæ»ú»¯³É¹¦£¬½á¹ûÐÎ×´: {x_null2.shape}")
        
    except ValueError as e:
        print(f"¾àÀë·½·¨Ò²Ê§°Ü: {e}")
'''
