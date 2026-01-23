import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =========================
# 全局风格
# =========================
plt.rcParams["font.family"] = "Arial"

# =========================
# 1. 读取并解析 txt
# =========================
pattern = re.compile(
    r"\[([a-zA-Z]+), comp (\d+), ([a-zA-Z_]+)\]\s+BEST ACC=([0-9.]+),"
)
# [male, comp 2, ANG] BEST ACC=0.5643, C=0.0005, down=0

rows = []
rows_null = []

with open("./source_files/sex_classification/LSVM_volume_cerebellum.txt", "r") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        
        sex = m.group(1)
        comp = int(m.group(2))
        patt = m.group(3)          # e.g. full_cerebellum_srp
        acc = float(m.group(4))

        if patt.endswith("REAL"):
            region = patt[:-5]     # 去掉 _real
            data_type = "real"
        elif patt.endswith("ANG"):
            region = patt[:-4]     # 去掉 _srp
            data_type = "srp"
        else:
            continue

        rows.append({
            "sex": sex,
            "comp": comp,
            "type": data_type,
            "acc": acc
        })
with open("./source_files/sex_classification/LSVM_volume_cerebellum_null.txt", "r") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        
        sex = m.group(1)
        comp = int(m.group(2))
        patt = m.group(3)          # e.g. full_cerebellum_srp
        acc = float(m.group(4))

        if patt.endswith("REAL"):
            region = patt[:-5]     # 去掉 _real
            data_type = "real"
        elif patt.endswith("ANG"):
            region = patt[:-4]     # 去掉 _srp
            data_type = "srp"
        else:
            continue

        rows_null.append({
            "sex": sex,
            "comp": comp,
            "type": data_type,
            "acc": acc
        })

df = pd.DataFrame(rows)
df_null = pd.DataFrame(rows_null)

# =========================
# 2. 聚合：mean ± std
# =========================
agg_df = (
    df
    .groupby(["sex", "comp", "type"])
    .agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        n_runs=("acc", "count")
    )
    .reset_index()
)

# =========================
# 3. 画图参数
# =========================
sex_order = [
    "male",
    "female",
]

sex_label_map = {
    "male": "Classification of top and bottom volume subgroups (Male)",
    "female": "Classification of top and bottom volume subgroups (Female)"
}

comp_order = [1, 2, 3]
type_order = ["real", "srp"]
pattern_order = ["Complex", "SRP"]

colors = [
    (51/256,  51/256, 255/256), 
    (255/256, 1/256,  1/256)
]
colors = ['#ed5f5f', '#53a8e1']

bar_width = 0.1
group_gap = 0.4

n_groups = len(comp_order) * len(type_order)
group_x = np.arange(n_groups) * (len(sex_order) * bar_width + group_gap)

# =========================
# 4. 绘图
# =========================
def plot_density_scatter(ax, group_data, x_center, color, sex, jitter_scale=0.1):
    kde = gaussian_kde(group_data)
    density = kde(group_data)
    
    # 根据密度调整偏移
    max_density = max(density)
    offsets = (density / max_density) * jitter_scale  # 偏移与密度成比例
    
    # 散点水平位置
    #x_positions = x_center + np.random.choice([-1, -1], size=len(group_data)) * offsets
    if sex==0:
        x_positions = x_center - 0.055 + np.random.uniform(-1.5, 0, size=len(group_data)) * offsets
    else:
        x_positions = x_center + 0.055 + np.random.uniform(0, 1.5, size=len(group_data)) * offsets
    
    # 绘制散点
    if po == 5:
        ax.scatter(x_positions, group_data, alpha=0.5, s=3, color=color, label='Permutation model')
    else:
        ax.scatter(x_positions, group_data, alpha=0.5, s=3, color=color)

fig, ax = plt.subplots(figsize=(5.2, 3), dpi=300)

for i, sex in enumerate(sex_order):
    means = []
    nulls = []

    for comp in comp_order:
        for t in type_order:
            row = agg_df[
                (agg_df["sex"] == sex) &
                (agg_df["comp"] == comp) &
                (agg_df["type"] == t)
            ]
            row_null = df_null[
                (df_null["sex"] == sex) &
                (df_null["comp"] == comp) &
                (df_null["type"] == t)
            ]

            means.append(row["acc_mean"].values[0])
            nulls.append(np.array(row_null['acc']))

    x_pos = group_x + i * bar_width

    # 柱子
    print(means)
    for value in range(len(means)):
        values_up = 1 - len(nulls[value][nulls[value]<means[value]])/1000
        values_down = 1 - len(nulls[value][nulls[value]>means[value]])/1000
        print(min(values_up, values_down))
    ax.bar(
        x_pos,
        means,
        bar_width,
        facecolor=colors[i],          # 中间填充白色
        edgecolor=colors[i],        # 边框颜色
        linewidth=0, 
        label=sex_label_map[sex]
    )
    for po in range(len(x_pos)):
        plot_density_scatter(ax, nulls[po], x_pos[po], colors[i], i)


# =========================
# 5. 坐标轴 & 美化
# =========================
x_labels = [f"{t} PC{c}" for c in comp_order for t in pattern_order]
ax.set_xticks(group_x + (len(sex_order) - 1) * bar_width / 2)
plt.axhline(0.5, color='gray', linestyle='dashed')

ax.set_xticks(group_x + (len(sex_order) - 1) * bar_width / 2)
ax.set_xticklabels(x_labels, fontsize=8)

ax.set_xlabel("Pattern", fontsize=8)
ax.set_ylabel("Accuracy", fontsize=8)
ax.set_ylim(0.3, 0.8)

ax.tick_params(axis="both", which="major", labelsize=8)

# 去掉上、右框线
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
order = [2, 3, 0, 1]
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    ncol=2,
    fontsize=7,
    loc="upper left",
    frameon=False,
    columnspacing=1.0,
    labelspacing=0.4
)
plt.tight_layout()

plt.savefig("./output/sex_classification/sex_volume", bbox_inches='tight')

plt.show()


