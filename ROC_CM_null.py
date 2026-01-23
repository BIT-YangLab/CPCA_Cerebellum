import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, auc, roc_curve, roc_auc_score
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import glob

plt.rcParams['font.family'] = 'Arial'

def plot_ROC_curve(df, figsize=(3, 3)):
    # 计算 ROC 曲线和 AUC
    fpr, tpr, _ = roc_curve(df['TrueLabel'], df['Prob'])
    auc_val = roc_auc_score(df['TrueLabel'], df['Prob'])
    
    # 构建结果字典，供你原始绘图代码使用
    results = {
        'model': {
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_val
        }
    }
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    style_dict = {
        'model': {'color': 'red', 'linestyle': '-', 'label': None}
    }
    for name, res in results.items():
        auc_val = res['auc']
        style = style_dict[name]
        style['label'] = f"{name} (AUC = {auc_val:.2f})"
        ax.plot(res['fpr'], res['tpr'], lw=2, **style)
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity", fontsize=8)
    ax.set_ylabel("Sensitivity", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.025, 0.95, f'AUC = {auc_val:.2f}', fontsize=8)
    inset_ax = fig.add_axes([0.62, 0.23, 0.3, 0.3])
    sns.kdeplot(df['Prob'][df['TrueLabel'] == 0], label='Female', fill=True, color='purple')
    sns.kdeplot(df['Prob'][df['TrueLabel'] == 1], label='Male', fill=True, color='orange')
    ymin, ymax = inset_ax.get_ylim()
    inset_ax.set_xlabel("", fontsize=8)
    inset_ax.set_ylabel("", fontsize=8)
    inset_ax.set_xlabel("Predicted Probability", fontsize=8)
    inset_ax.set_ylabel("Density", fontsize=8)
    inset_ax.set_xlim(0, 1)
    inset_ax.tick_params(axis='both', labelsize=8)
    inset_ax.tick_params(axis='both', labelsize=8)
    inset_ax.set_xticks([])
    # inset_ax.text(0.7, 0.88*ymax, 'Male', color='orange', fontsize=8)
    # inset_ax.text(0.02, 0.88*ymax, 'Female', color='purple', fontsize=8)
    # inset_ax.legend(loc='lower right', fontsize=8)
    # ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"./output/sex_classification/ROC_{patt}_{model}_{comp+1}", dpi=300)
    # plt.show()

def plot_confusion_matrix_stats(y_true, y_pred, comp, model, patt, labels=['Female', 'Male'], figsize=(3, 1.5)):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    chi2, p, dof, expected = chi2_contingency(cm)
    fig, (ax, axt) = plt.subplots(1, 2, figsize=figsize, dpi=300)
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap='inferno', cbar=False, 
                vmin=0, vmax=1, 
                xticklabels=labels, yticklabels=labels, square=True,
                linewidths=1, linecolor='white', ax=ax, annot_kws={"size": 8})
    ax.set_xlabel('Predicted label', fontsize=8)
    ax.set_ylabel('True label', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    if p < 0.001:
        stats_text = (
            f"$\chi^2$ = {chi2:.2f}\n$"
            r"P"f"$ < 0.001"
        )
    else:
        stats_text = (
            f"$\chi^2$ = {chi2:.2f}\n$"
            r"P"f"$ = {p:.4f}"
        )
    axt.set_axis_off()
    axt.text(0, 0.5, stats_text, va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"./output/sex_classification/cm_{patt}_{model}_{comp+1}", bbox_inches='tight')
    # plt.show()

def plot_null_model(y_true, y_pred, patt, model, comp, array, padding=0.05, figsize=(2.25, 0.75)):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    accu = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.histplot(array[comp, :], bins=30, kde=True, color='gray')
    p_value = np.mean(array[comp, :] >= accu)
    if p_value < 0.001:
        p = ' < 0.001'
    else:
        p = f' = {p_value:.3f}'
    x_min = max(0, float(f'{min(array[comp, :])}'[:3])-0.1-padding)
    x_max = max(accu, max(array[comp, :]))
    _, ymax = plt.gca().get_ylim()
    plt.axvline(accu, color='red', linestyle='dashed')
    plt.xticks(np.arange(x_min, accu, 0.1))
    plt.xticks(np.arange(0, accu, 0.1))
    plt.xlabel('Accuracy', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8)
    # ax.text(x_min+0.1*(accu-x_min), 0.7*ymax, 
    #         f'Accuracy = {accu:.3f}\n$'r'P'f'${p}', size=8)
    ax.text(0.1*(x_max), 0.7*ymax, 
            f'Accuracy = {accu:.3f}\n$'r'P'f'${p}', size=8)
    plt.savefig(f"./output/sex_classification/Null_model_{patt}_{model}_{comp}", bbox_inches='tight')
    # plt.show()

for model in ['SVM', 'ANN']:
    for patt in ['real', 'srp']:
        for comp in range(3):
            # 读取 CSV 文件
            path = "./source_files/sex_classification"

            pattern_csv = f"Final_comp{comp+1}_cere_{patt}_{model}_*_predictions.csv"
            files = glob.glob(os.path.join(path, pattern_csv))
            df = pd.read_csv(files[0])

            pattern_txt = f"null_{patt}_{model}.txt"
            
            with open(os.path.join(path, pattern_txt), 'r') as f:
                lines = f.readlines()
            data = np.array([list(map(float, line.strip().split(', '))) for line in lines])
            
            plot_ROC_curve(df)

            y_true = df['TrueLabel']
            y_score = df['Prob']
            
            thresholds = np.linspace(0, 1, 100)
            f1s = []
            accs = []
            
            for t in thresholds:
                preds = (y_score >= t).astype(int)
                f1s.append(f1_score(y_true, preds))
                accs.append(accuracy_score(y_true, preds))
                
            y_true, y_pred = df['TrueLabel'], df['PredictedLabel']
            
            plot_confusion_matrix_stats(y_true, y_pred, comp, model, patt)

            plot_null_model(y_true, y_pred, patt, model, comp, data)