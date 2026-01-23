'''
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
pattern = re.compile(
    r"\[comp (\d+), ([a-zA-Z_]+)\] "
    r"BEST ACC=([0-9.]+), C=([0-9.eE-]+), down=(\d+)"
)

rows = []

with open("LSVM_sex_wholebrain.txt", "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            comp = int(m.group(1))
            region_type = m.group(2)
            acc = float(m.group(3))
            C = float(m.group(4))
            down = int(m.group(5))

            # 拆 region / type
            if region_type.endswith("_Complex"):
                region = region_type.replace("_Complex", "")
                data_type = "Complex"
            elif region_type.endswith("_SRP"):
                region = region_type.replace("_SRP", "")
                data_type = "SRP"
            else:
                region = region_type
                data_type = "unknown"

            rows.append({
                "comp": comp,
                "region": region,
                "type": data_type,
                "acc": acc,
                "C": C,
                "down": down
            })

df = pd.DataFrame(rows)
mean = df.groupby(["region", "type"])["acc"].mean()
std = df.groupby(["region", "type"])["acc"].std()


df_extend = pd.read_csv("LSVM_sex_wholebrain_simulated.csv")
#df_extend = df_extend.iloc[:50, :]

agg_mean_std = (
    df_extend
    .groupby(["comp", "region_type"])
    .agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        n_runs=("acc", "count")
    )
    .reset_index()
)
agg_df = agg_mean_std.copy()

def split_region_type(rt):
    if rt.endswith("_Complex"):
        return rt[:-8], "Complex"
    elif rt.endswith("_SRP"):
        return rt[:-4], "SRP"
    else:
        return rt, "unknown"

agg_df[["region", "type"]] = agg_df["region_type"].apply(
    lambda x: pd.Series(split_region_type(x))
)

region_order = ["Cerebellum", "Cortex", "Whole-brain", "Whole-brain cerebellum"]
comp_order = [1, 2, 3]
type_order = ["Complex", "SRP"]
colors = [(237/256, 95/256, 95/256), 
          (83/256, 168/256, 225/256), 
          (154/256, 114/256, 199/256), 
          (208/256, 187/256, 219/256)]

# =========================
# 参数
# =========================
bar_width = 0.18
group_gap = 0.25

# 6 个主刻度的位置
n_groups = len(comp_order) * len(type_order)
group_x = np.arange(n_groups) * (4 * bar_width + group_gap)

fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

# =========================
# 画柱子
# =========================
for i, region in enumerate(region_order):
    means = []
    stds = []
    for comp in comp_order:
        for t in type_order:
            row = agg_df[
                (agg_df["comp"] == comp) &
                (agg_df["type"] == t) &
                (agg_df["region"] == region)
            ]

            means.append(row["acc_mean"].values[0])
            stds.append(row["acc_std"].values[0])

    x_pos = group_x + i * bar_width

    ax.bar(
        x_pos,
        means,
        bar_width,
        capsize=4,
        label=region, 
        color=colors[i]
    )
    ax.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=2.5,
        capthick=1
    )

# =========================
# x 轴刻度
# =========================
x_labels = []
for comp in comp_order:
    for t in type_order:
        x_labels.append(f"{t}\nPC{comp}")

ax.set_xticks(group_x + 1.5 * bar_width)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_xlabel("Pattern", fontsize=8)

# =========================
# 美化
# =========================
ax.set_ylabel("Accuracy", fontsize=8)
ax.set_ylim(0.5, 1)
ax.legend(fontsize=8)
#ax.set_title("Classification Accuracy (mean ± std)")

ax.tick_params(axis='both', which='major', labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
'''

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 全局风格
# =========================
plt.rcParams["font.family"] = "Arial"

# =========================
# 1. 读取并解析 txt
# =========================
pattern = re.compile(
    r"\[comp (\d+), ([a-zA-Z_]+)\]\s+BEST ACC=([0-9.]+),"
)

rows = []

with open("./source_files/sex_classification/LSVM_sex_wholebrain.txt", "r") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue

        comp = int(m.group(1))
        region_type = m.group(2)          # e.g. full_cerebellum_srp
        acc = float(m.group(3))

        if region_type.endswith("_real"):
            region = region_type[:-5]     # 去掉 _real
            data_type = "real"
        elif region_type.endswith("_srp"):
            region = region_type[:-4]     # 去掉 _srp
            data_type = "srp"
        else:
            continue

        rows.append({
            "comp": comp,
            "region": region,
            "type": data_type,
            "acc": acc
        })

df = pd.DataFrame(rows)

# =========================
# 2. 聚合：mean ± std
# =========================
agg_df = (
    df
    .groupby(["comp", "region", "type"])
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
region_order = [
    "cere",
    "cortex",
    "full",
    "full_cerebellum"
]

region_label_map = {
    "cere": "Cerebellum",
    "cortex": "Cortex",
    "full": "Whole-brain",
    "full_cerebellum": "Whole-brain cerebellum"
}

comp_order = [1, 2, 3]
type_order = ["real", "srp"]
pattern_order = ["Complex", "SRP"]

colors = [
    (237/256, 95/256, 95/256),     # Cerebellum
    (83/256, 168/256, 225/256),    # Cortex
    (154/256, 114/256, 199/256),   # Whole-brain
    (208/256, 187/256, 219/256)    # Whole-brain cerebellum
]

bar_width = 0.18
group_gap = 0.25

n_groups = len(comp_order) * len(type_order)
group_x = np.arange(n_groups) * (len(region_order) * bar_width + group_gap)

from scipy.stats import ttest_rel

print("Paired t-test: Cerebellum vs Cortex (within each comp × type)\n")

for comp in comp_order:
    for t in type_order:
        cere_vals = df[
            (df["comp"] == comp) &
            (df["type"] == t) &
            (df["region"] == "cere")
        ]["acc"].values

        cortex_vals = df[
            (df["comp"] == comp) &
            (df["type"] == t) &
            (df["region"] == "cortex")
        ]["acc"].values
        t_stat, p_val = ttest_rel(cere_vals, cortex_vals)
        print(
            f"c {comp} | {t.upper()} : "
            f"Cerebellum vs Cortex | "
            f"mean_cere={cere_vals.mean():.4f}, "
            f"mean_cortex={cortex_vals.mean():.4f}, "
            f"t={t_stat:.3f}, p={p_val:.4g}"
        )

        cortex_vals = df[
            (df["comp"] == comp) &
            (df["type"] == t) &
            (df["region"] == "full")
        ]["acc"].values
        t_stat, p_val = ttest_rel(cere_vals, cortex_vals)
        print(
            f"c {comp} | {t.upper()} : "
            f"Cerebellum vs brain | "
            f"mean_cerebellum={cere_vals.mean():.4f}, "
            f"mean_brain={cortex_vals.mean():.4f}, "
            f"t={t_stat:.3f}, p={p_val:.4g}"
        )

        cortex_vals = df[
            (df["comp"] == comp) &
            (df["type"] == t) &
            (df["region"] == "full_cerebellum")
        ]["acc"].values
        t_stat, p_val = ttest_rel(cere_vals, cortex_vals)
        print(
            f"c {comp} | {t.upper()} : "
            f"Cerebellum vs brain cere | "
            f"mean_cerebellum={cere_vals.mean():.4f}, "
            f"mean_brain_cere={cortex_vals.mean():.4f}, "
            f"t={t_stat:.3f}, p={p_val:.4g}"
        )


# =========================
# 4. 绘图
# =========================
fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

for i, region in enumerate(region_order):
    means, stds = [], []

    for comp in comp_order:
        for t in type_order:
            row = agg_df[
                (agg_df["comp"] == comp) &
                (agg_df["type"] == t) &
                (agg_df["region"] == region)
            ]

            means.append(row["acc_mean"].values[0])
            stds.append(row["acc_std"].values[0])

    x_pos = group_x + i * bar_width

    # 柱子
    ax.bar(
        x_pos,
        means,
        bar_width,
        color=colors[i],
        label=region_label_map[region]
    )

    # 误差棒（细）
    ax.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2,
        capthick=0.8
    )

# =========================
# 5. 坐标轴 & 美化
# =========================
x_labels = [f"{t} PC{c}" for c in comp_order for t in pattern_order]

ax.set_xticks(group_x + (len(region_order) - 1) * bar_width / 2)
ax.set_xticklabels(x_labels, fontsize=8)

ax.set_xlabel("Pattern", fontsize=8)
ax.set_ylabel("Accuracy", fontsize=8)
ax.set_ylim(0.5, 1.0)

ax.tick_params(axis="both", which="major", labelsize=8)

# 去掉上、右框线
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=7, loc="upper center", frameon=False)

plt.tight_layout()

plt.savefig(f"./output/sex_classification/sex_wholebrain", bbox_inches='tight')

# plt.show()


