import numpy as np
if not hasattr(np, "MachAr"):
    class MachAr:
        def __init__(self, *args, **kwargs):
            self.eps = np.finfo(float).eps
    np.MachAr = MachAr
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.svm import SVC
import glob
import pandas as pd
import os
import nibabel as nib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams['font.family'] = 'Arial'

# Configuration
target_directory = '/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_cere/CPCA_HCP_100_SRP_all'
target_directory2 = '/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_cere/CPCA_HCP_Real_all'
target_directory_cortex = '/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_cortex/CPCA_HCP_100_SRP_all'
target_directory2_cortex = '/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_cortex/CPCA_HCP_100_real_all'
target_directory_full = '/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_full/CPCA_HCP_100_SRP_all'
target_directory2_full = '/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_full/CPCA_HCP_100_real_all'
save_dictionary = 'results_svm'
file_extension = 'nii'
threshold = 0.05
comps = 3
voxels_all = 91282
voxels = 17853
cortex_e = 64984
cere_s, cere_e = 65289, 83142
num_subj = 140



def read_files_with_extension(directory, id_txt_path, keyword1, keyword2, region):
    '''
    Load data
    '''
    def pull_cifti_data(cifti_obj):
        cifti_obj.set_data_dtype('<f4')
        cifti_data = np.array(cifti_obj.get_fdata())
        n_time = cifti_data.shape[0]
        return cifti_obj.header, cifti_data, n_time
    with open(id_txt_path, 'r') as file:
        subject_ids = np.array(sorted([int(line.strip()) for line in file.readlines()]))
    cifti_maps_all = []
    header = None
    for subject_id in subject_ids:
        # pattern = os.path.join(directory, f"aligned_pca*{subject_id}*{keyword1}*{keyword2}*.nii")
        pattern = os.path.join(directory, f"*{subject_id}*{keyword1}*{keyword2}*.nii")
        # if not os.path.isfile(f'/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/results_cere/CPCA_HCP_100_CRP_all/srp_cere_pca_cere_m_{subject_id}_REST2_smooth_norm_filt_complex_ang_results_results.dtseries.nii'):
        #     continue
        matched_files = sorted(glob.glob(pattern))
        #print(matched_files)
        for file_path in matched_files:
            if "align" not in file_path and ("srp" in file_path or "real" in file_path):
                hdr, cifti_maps, n_time = pull_cifti_data(nib.load(file_path))
                if region == 'cortex':
                    cifti_maps_all.append(cifti_maps[:comps, :cortex_e])
                if region == 'cerebellum':
                    cifti_maps_all.append(cifti_maps[:comps, cere_s:cere_e])
                if region == 'full':
                    cifti_maps_all.append(np.concatenate((cifti_maps[:comps, :cortex_e], cifti_maps[:comps, cere_s:cere_e]), axis=1))
                if header is None:
                    header = hdr
    
    return cifti_maps_all, header

def regress_covariates(files_f, filtered_df):
    # covariates = filtered_df[['Age_in_Yrs', 'L_WM_Vol', 'R_WM_Vol', 'L_GM_Vol', 'R_GM_Vol']].copy()
    # numeric_cols = ['Age_in_Yrs', 'L_WM_Vol', 'R_WM_Vol', 'L_GM_Vol', 'R_GM_Vol']
    
    # covariates = filtered_df[['L_WM_Vol', 'R_WM_Vol', 'L_GM_Vol', 'R_GM_Vol']].copy()
    # # numeric_cols = ['L_WM_Vol', 'R_WM_Vol', 'L_GM_Vol', 'R_GM_Vol']
    # covariates[numeric_cols] = StandardScaler().fit_transform(covariates[numeric_cols])
    if 'WM_total' not in filtered_df.columns:
        filtered_df['WM_total'] = filtered_df['L_WM_Vol'] + filtered_df['R_WM_Vol']
        filtered_df['GM_total'] = filtered_df['L_GM_Vol'] + filtered_df['R_GM_Vol']
    covariates = filtered_df[['WM_total', 'GM_total']].copy()
    # covariates[['WM_total', 'GM_total']] = StandardScaler().fit_transform(covariates[['WM_total', 'GM_total']])
    X_cov = add_constant(covariates.values)
    assert files_f.shape[0] == X_cov.shape[0], \
        f"files_f ÓÐ {files_f.shape[0]} ¸ö±»ÊÔ£¬µ« covariates ÓÐ {X_cov.shape[0]} ÐÐ£¬Ñù±¾Êý²»Ò»ÖÂ£¡"
    model = OLS(files_f, X_cov)
    results = model.fit()
    return results.resid

def feature_select(X, y, k):
    var = X.var(axis=0)
    non_const_mask = var > 0
    if not np.any(non_const_mask):
        raise ValueError("ËùÓÐÌØÕ÷¶¼ÊÇ³£Êý£¨»ò·½²îÌ«Ð¡£©£¬ÎÞ·¨×öÌØÕ÷Ñ¡Ôñ")
    X_nc = X[:, non_const_mask]
    k = min(k, X_nc.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_sel = selector.fit_transform(X, y)
    idx = selector.get_support(indices=True)
    return X_sel, idx

def evaluate_kfold(model, X, y, repeat, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=114+repeat)
    scores = cross_val_score(model, X, y, cv=skf)
    return scores.mean(), scores

def evaluate_holdout(model, X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=114
    )
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def nested_svm_cv(X, y, k_ratios, C_list, n_outer=5, n_inner=5, random_state=114):
    """
    Nested CV for linear SVM with feature selection
    """
    outer_cv = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=n_inner, shuffle=True, random_state=random_state
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
        ("svm", SVC(kernel="linear"))
    ])

    param_grid = {
        "select__k": k_ratios,
        "svm__C": C_list
    }

    outer_scores = []
    best_params_all = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        outer_scores.append(acc)
        best_params_all.append(grid.best_params_)

    return (
        np.mean(outer_scores),
        np.std(outer_scores),
        best_params_all
    )

file_list = '/home/lvshuo/VDisk4/Lvshuo/visual_module/output_Sex.txt'
names = []
with open(file_list, 'r') as file:
    lines = file.readlines()
for i in lines:
    names.append(int(i[:6]))
names.sort()
# print(len(names), names)
# Label loading and encoding
df = pd.read_csv('/home/lvshuo/VDisk4/Lvshuo/2024_cerebellum_CPCA_rs-patterns_in_HCP/behavior/PCA_35_info_final.csv')
subj_col = df.columns[0]
filtered_df = df[df[subj_col].isin(names)].copy()
filtered_df = filtered_df.sort_values(by=subj_col)
filtered_df['WM_total'] = filtered_df['L_WM_Vol'] + filtered_df['R_WM_Vol']
filtered_df['GM_total'] = filtered_df['L_GM_Vol'] + filtered_df['R_GM_Vol']
ids_from_df = filtered_df[subj_col].to_numpy()
names_sorted = np.array(sorted(names))
assert np.array_equal(ids_from_df, names_sorted)
Y_sex = filtered_df['Gender'].values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(Y_sex)

# Data loading
files_real_full, _ = read_files_with_extension(target_directory2_full, file_list, 'REST2', 'real', 'full')
files_srp_full, _ = read_files_with_extension(target_directory_full, file_list, 'REST2', 'ang', 'full')
files_real_cortex, _ = read_files_with_extension(target_directory2_cortex, file_list, 'REST2', 'real', 'cortex')
files_srp_cortex, _ = read_files_with_extension(target_directory_cortex, file_list, 'REST2', 'ang', 'cortex')
files_real_full_cerebellum, _ = read_files_with_extension(target_directory2_full, file_list, 'REST2', 'real', 'cerebellum')
files_srp_full_cerebellum, _ = read_files_with_extension(target_directory_full, file_list, 'REST2', 'ang', 'cerebellum')
files_real_cerebellum, hdr = read_files_with_extension(target_directory2, file_list, 'REST2', 'real', 'cerebellum')
files_srp_cerebellum, _ = read_files_with_extension(target_directory, file_list, 'REST2', 'ang', 'cerebellum')

# Results initailization
pattern = ["cere_real", "cere_srp", "cortex_real", "cortex_srp", "full_real", "full_srp", "full_cerebellum_real", "full_cerebellum_srp"]
weights = [np.zeros((comps, voxels_all)) for i in range(len(pattern))]

selected_features = [np.zeros((comps, voxels_all)) for i in range(len(pattern))]

for repeat in range(5, 5+32):
    for comp in range(comps):
        files = [np.array(files_real_cerebellum)[:, comp, :], np.array(files_srp_cerebellum)[:, comp, :], 
                np.array(files_real_cortex)[:, comp, :], np.array(files_srp_cortex)[:, comp, :], 
                np.array(files_real_full)[:, comp, :], np.array(files_srp_full)[:, comp, :], 
                np.array(files_real_full_cerebellum)[:, comp, :], np.array(files_srp_full_cerebellum)[:, comp, :]]
        for f in range(len(files)):
            #X = regress_covariates(files[f], filtered_df)
            X = files[f]
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            best_mean_acc = -np.inf
            best_holdout_acc = None
            best_c = None
            best_down = None
            best_selected_indices = None
            for c in range(1, 11):
                c_para = 0.0001 * c
                for down in range(5):
                    k_ratio = (down + 1) / 5
                    select_voxels = int(k_ratio * X.shape[1])
                    #select_voxels = voxels*((down+1)/5)
                    svm_model = SVC(kernel='linear', C=c_para)
                    #svm_model = LinearSVC(C=1000*c_para, penalty='l1', dual='auto')
                    X_sel, selected_indices = feature_select(X, labels, k=int(select_voxels))
                    mean_acc, all_acc = evaluate_kfold(svm_model, X_sel, labels, repeat, 5)
                    # print("10-fold:", mean_acc)
                    # print("80/20:", holdout_acc)
                    
                    
                    if mean_acc > best_mean_acc:
                        best_mean_acc = mean_acc
                        best_c = c_para
                        best_down = down
                        best_selected_indices = selected_indices
            X_best = X[:, best_selected_indices]
            
            svm_best = SVC(kernel='linear', C=best_c)
            #svm_best = LinearSVC(C=1000*c_para, penalty='l1', dual='auto')
            svm_best.fit(X_best, labels)
            w = svm_best.coef_.flatten()
            if f in [0, 1, 6, 7]:
                weights[f][comp, cere_s:cere_e] = 0
                weights[f][comp, best_selected_indices + cere_s] = w
                selected_features[f][comp, cere_s:cere_e] = 0
                selected_features[f][comp, best_selected_indices + cere_s] = 1
            if f in [2, 3]:
                weights[f][comp, :cortex_e] = 0
                weights[f][comp, best_selected_indices] = w
                selected_features[f][comp, :cortex_e] = 0
                selected_features[f][comp, best_selected_indices] = 1
            #if f in [4, 5]:
                #weights[f][comp, :cortex_e] = 0
                #weights[f][comp, best_selected_indices + cere_s] = w
                #weights[f][comp, cere_s:cere_e] = 0
                #weights[f][comp, best_selected_indices + cere_s] = w
                #selected_features[f][comp, cere_s:cere_e] = 0
                #selected_features[f][comp, best_selected_indices + cere_s] = 1
                #selected_features[f][comp, cere_s:cere_e] = 0
                #selected_features[f][comp, best_selected_indices + cere_s] = 1
            
            print(
                f"[comp {comp+1}, {pattern[f]}] BEST ACC={best_mean_acc:.4f}, "
                f"C={best_c}, down={best_down}"
            )
    #for i in range(2):
        #modified_cifti_img = nib.Cifti2Image(weights[i], hdr)
        #nib.save(modified_cifti_img, f'/home/lvshuo/VDisk4/Lvshuo/visual_module/1228_weights_{pattern[i]}_{repeat}.dtseries.{file_extension}')
        #modified_cifti_img = nib.Cifti2Image(selected_features[i], hdr)
        #nib.save(modified_cifti_img, f'/home/lvshuo/VDisk4/Lvshuo/visual_module/1228_features_{pattern[i]}_{repeat}.dtseries.{file_extension}')

        





# Specific model

c_list = [[0.0004, 0.0001, 0.0001], 
          [0.0001, 0.0004, 0.0005]]
down_list = [[0, 0, 0], 
             [0, 1, 0]]
             
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def compute_stable_feature_mask(X, y, down):
    """
    One-time feature ranking to define a stable feature space
    (NO model training, NO performance estimation)
    """
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    k_ratio = (down + 1) / 5
    k_features = int(k_ratio * X.shape[1])

    selector = SelectKBest(f_classif, k=k_features)
    selector.fit(Xz, y)

    return selector.get_support(indices=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import numpy as np


def run_specific_model_cv_stableFS(
    X,
    y,
    best_c,
    feature_idx,
    model_type="SVM",
    n_splits=5,
    random_state=114
):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    all_true, all_pred, all_prob = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train[:, feature_idx])
        X_test  = scaler.transform(X_test[:, feature_idx])

        if model_type == "SVM":
            model = SVC(
                kernel="linear",
                C=best_c,
                probability=True
            )
        elif model_type == "ANN":
            model = MLPClassifier(
                hidden_layer_sizes=(20,),
                max_iter=1000,
                random_state=random_state
            )
        else:
            raise ValueError("model_type must be 'SVM' or 'ANN'")

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        all_true.extend(y_test)
        all_pred.extend(pred)
        all_prob.extend(prob)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)
    fpr, tpr, _ = roc_curve(all_true, all_prob)
    auc_score = auc(fpr, tpr)

    return {
        "acc": acc,
        "auc": auc_score,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_true": all_true,
        "y_pred": all_pred,
        "y_prob": all_prob
    }

import os
import pandas as pd
import matplotlib.pyplot as plt

save_dir = "/home/lvshuo/VDisk4/Lvshuo/visual_module/specific_model_results"
os.makedirs(save_dir, exist_ok=True)

models = ["SVM", "ANN"]

for comp in range(comps):

    X_list = [
        np.array(files_real_cerebellum)[:, comp, :],
        np.array(files_srp_cerebellum)[:, comp, :]
    ]

    for f, X in enumerate(X_list):

        y = labels
        best_c = c_list[f][comp]
        best_down = down_list[f][comp]

        tag = f"comp{comp+1}_{pattern[f]}"

        print(f"\n[{tag}] Stable feature selection ...")

        # ===== ÎÈ¶¨ÌØÕ÷£¨Ö»ËãÒ»´Î£©=====
        feature_idx = compute_stable_feature_mask(X, y, best_down)

        for model_type in models:

            print(f"  Running {model_type} ...")

            results = run_specific_model_cv_stableFS(
                X=X,
                y=y,
                best_c=best_c,
                feature_idx=feature_idx,
                model_type=model_type
            )

            print(
                f"  RESULT: ACC={results['acc']:.4f}, "
                f"AUC={results['auc']:.4f}"
            )

            prefix = (
                f"{save_dir}/FINAL_{tag}_{model_type}"
                f"_{results['auc']:.4f}_C{best_c:.4f}_down{best_down}"
            )

            # ===== ±£´æÔ¤²â½á¹û =====
            df_pred = pd.DataFrame({
                "TrueLabel": results["y_true"],
                "PredictedLabel": results["y_pred"],
                "Prob": results["y_prob"]
            })
            df_pred.to_csv(f"{prefix}_predictions.csv", index=False)

            # ===== ±£´æ»ìÏý¾ØÕó =====
            np.savetxt(
                f"{prefix}_confusion_matrix.txt",
                results["cm"],
                fmt="%d"
            )

            # ===== ±£´æÖ¸±ê =====
            with open(f"{prefix}_metrics.txt", "w") as ftxt:
                ftxt.write(f"ACC\t{results['acc']:.6f}\n")
                ftxt.write(f"AUC\t{results['auc']:.6f}\n")