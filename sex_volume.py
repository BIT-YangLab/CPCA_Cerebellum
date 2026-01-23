import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

file_list = 'D:/Desktop/2025_CPCA_Cerebellum/Code-Three/source_files/output_Sex.txt'
behavior_csv = 'D:/Desktop/2025_CPCA_Cerebellum/Code-Three/source_files/PCA_35_info_final.csv'

def get_top_bottom_33(df, value_col):
    n = len(df)
    k = max(1, int(n * 0.33))
    df_sorted = df.sort_values(value_col)
    bottom = df_sorted.iloc[:k]
    top = df_sorted.iloc[-k:]
    return top, bottom

def multi_violin_box_scatter(data, colors_7, tags_7):
    fig, ax= plt.subplots(figsize=(5, 3), dpi=300)
    for i in range(len(data)):
        parts = ax.violinplot(data[i], positions=[i], 
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            verts =pc.get_paths()[0].vertices
            verts[:, 0]=np.clip(verts[:, 0], i, np.inf)
            pc.set_facecolor(colors_7[i])
            pc.set_edgecolor('none')
            pc.set_alpha(1)
    box_positions = np.arange(len(data)) - 0.1
    bp = ax.boxplot(
        data,
        positions=box_positions,
        widths=0.2,
        patch_artist=False,
        showfliers=True,
        medianprops=dict(color='black', linewidth=1),  # 中位线保留
        whiskerprops=dict(linewidth=0),                # 去掉
        capprops=dict(linewidth=0),                    # 去掉
        boxprops=dict(linewidth=0),                    # 去掉
        flierprops=dict(marker='o', color='black', markersize=1)
    )
    for box, color in zip(bp['boxes'], colors_7):
        box.set(color="black", linewidth=1, alpha=1)
        #box.set_edgecolor(color='black')
        #box.set_facecolor(color='white')
    for i, d in enumerate(data):
        jittered_x= np.random.normal(loc=i-0.1, scale=0.025, size=len(d))
        ax.scatter(jittered_x, d, color=colors_7[i], alpha=0.8, s=5)
    for group in [0, 2, 4]:
        t_stat, p_value = ttest_ind(data[group], data[group+1])
        print(t_stat, p_value)
        if p_value < 0.05:
            y, h, col = max([data[group].max(), data[group+1].max()]), 0.01, 'k'
            plt.plot([group, group, group+1, group+1], [y+1200, y+1200+h, y+1200+h, y+1200], lw=1.5, c=col)
            sig = '*'
            if p_value < 0.01:
                sig = '**'
                if p_value < 0.001:
                    sig = '***'
            plt.text(group+0.5, y-1000, sig, ha='center', va='bottom', color=col, fontsize=10)
    
    ax.set_xticks(np.arange(len(tags_7)))
    ax.set_xticklabels(tags_7, fontsize=8)
    ax.set_ylabel('Total Cerebellum Volume', fontsize=8)
    # ax.yaxis.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    # ax.set_title("%sComp%d"%(title_list[pattern], comp), fontsize=20)
    plt.savefig(f"./output/sex_classification/volume_differences", bbox_inches='tight')
    # plt.show()

names = []
with open(file_list, 'r') as file:
    lines = file.readlines()
for i in lines:
    names.append(int(i[:6]))
names.sort()
names = np.array(names)
df = pd.read_csv(behavior_csv)
subj_col = df.columns[0]

filtered_df = df[df[subj_col].isin(names)].copy()
filtered_df = filtered_df.sort_values(by=subj_col)

filtered_df['WM_total'] = filtered_df['L_WM_Vol'] + filtered_df['R_WM_Vol']
filtered_df['GM_total'] = filtered_df['L_GM_Vol'] + filtered_df['R_GM_Vol']
filtered_df['GM_WM_total'] = filtered_df['GM_total'] + filtered_df['WM_total']

df_male = filtered_df[filtered_df['Gender'] == 'M'].copy()
df_female = filtered_df[filtered_df['Gender'] == 'F'].copy()

male_top, male_bottom = get_top_bottom_33(df_male, 'GM_WM_total')
female_top, female_bottom = get_top_bottom_33(df_female, 'GM_WM_total')

data = [np.array(df_male["GM_WM_total"]), np.array(df_female["GM_WM_total"]),
        np.array(male_top["GM_WM_total"]), np.array(male_bottom["GM_WM_total"]), 
        np.array(female_top["GM_WM_total"]), np.array(female_bottom["GM_WM_total"])]

colors_7 = ['#ed5f5f', 
            '#53a8e1', 
            '#ed5f5f', 
            '#ed5f5f', 
            '#53a8e1', 
            '#53a8e1']
tags_7 = ['Male', 'Female', r'Male$_{top}$', r'Male$_{bottom}$', r'Female$_{top}$', r'Female$_{bottom}$']

multi_violin_box_scatter(data, colors_7, tags_7)
