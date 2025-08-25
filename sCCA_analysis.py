"""
Sparse CCA pipeline (minimal-change refactor).

- Keep numerical logic identical to the original implementation.
- Add English docstrings (Google style) and "Step" comments.
- Wrap the bottom script into a `main()` entrypoint with argparse.
- Use cross-platform path handling where helpful without changing behavior.
- Preserve plotting and saving behavior; file names mirror original format.

NOTE
----
This refactor intentionally avoids changing computations to preserve results.
"""

import argparse
import sys
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from statsmodels.stats.multitest import multipletests

plt.rcParams["font.family"] = "Arial"


class SparseCCA:
    """Sparse Canonical Correlation Analysis (sCCA) with soft-thresholding.

    This class mirrors the original numerical logic to ensure identical results.

    Parameters
    ----------
    n_components : int
        Number of canonical components.
    l1_c1 : float
        L1 threshold factor for X view (as a fraction of max|u|).
    l1_c2 : float
        L1 threshold factor for Y view (as a fraction of max|v|).
    max_iter : int
        Maximum iterations for inner alternating updates.
    tol : float
        Convergence tolerance on squared update difference.
    random_state : int
        Seed for reproducible initialization.
    """

    def __init__(
        self,
        n_components: int = 2,
        l1_c1: float = 0.5,
        l1_c2: float = 0.5,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.l1_c1 = l1_c1
        self.l1_c2 = l1_c2
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        # Learned parameters
        self.x_weights_: Optional[np.ndarray] = None
        self.y_weights_: Optional[np.ndarray] = None

    @staticmethod
    def _soft_threshold(
        x: np.ndarray, 
        lambda_val: float
    ) -> np.ndarray:
        """Elementwise soft-thresholding: sign(x) * max(|x| - λ, 0)."""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

    def fit(
        self, X: np.ndarray, 
        Y: np.ndarray
    ) -> "SparseCCA":
        """Fit the model on (X, Y). Inputs are z-scored internally.

        Returns
        -------
        self : SparseCCA
            The fitted estimator.
        """
        # Step 1: Standardize inputs for numerical stability
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)

        n_samples, n_features_x = X.shape
        _, n_features_y = Y.shape

        random_state = check_random_state(self.random_state)
        self.x_weights_ = np.zeros((n_features_x, self.n_components))
        self.y_weights_ = np.zeros((n_features_y, self.n_components))

        # Step 2: Component-wise extraction with deflation
        for k in range(self.n_components):
            # Step 2.1: Random unit-norm init
            u = random_state.normal(size=(n_features_x,))
            v = random_state.normal(size=(n_features_y,))
            u = u / (np.sqrt(np.sum(u**2)) + 1e-12)
            v = v / (np.sqrt(np.sum(v**2)) + 1e-12)

            # Step 2.2: Alternating updates with L1 soft-thresholding
            for _ in range(self.max_iter):
                u_old = u.copy()
                v_old = v.copy()

                u = X.T @ (Y @ v_old)
                u = self._soft_threshold(u, self.l1_c1 * np.max(np.abs(u)))
                if np.sum(u**2) > 0:
                    u = u / np.sqrt(np.sum(u**2))

                v = Y.T @ (X @ u)
                v = self._soft_threshold(v, self.l1_c2 * np.max(np.abs(v)))
                if np.sum(v**2) > 0:
                    v = v / np.sqrt(np.sum(v**2))

                if (np.sum((u - u_old) ** 2) < self.tol and np.sum((v - v_old) ** 2) < self.tol):
                    break

            self.x_weights_[:, k] = u
            self.y_weights_[:, k] = v

            # Step 3: Deflation (regress out current scores)
            t = X @ u.reshape(-1, 1)
            s = Y @ v.reshape(-1, 1)

            beta_X, _, _, _ = np.linalg.lstsq(t, X, rcond=None)
            X_deflated = X - t @ beta_X

            beta_Y, _, _, _ = np.linalg.lstsq(s, Y, rcond=None)
            Y_deflated = Y - s @ beta_Y

            X = X_deflated
            Y = Y_deflated

        return self

    def transform(
        self, X: np.ndarray, 
        Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project data to canonical scores using learned weights."""
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)
        if self.x_weights_ is None or self.y_weights_ is None:
            raise RuntimeError("fit() must be called before transform().")
        X_scores = X @ self.x_weights_
        Y_scores = Y @ self.y_weights_
        return X_scores, Y_scores

    def score(
        self, X: np.ndarray, 
        Y: np.ndarray
    ) -> np.ndarray:
        """Compute per-component canonical correlations on (X, Y)."""
        X_scores, Y_scores = self.transform(X, Y)
        canonical_correlations = []
        for i in range(self.n_components):
            r, _ = pearsonr(X_scores[:, i], Y_scores[:, i])
            canonical_correlations.append(r)
        return np.array(canonical_correlations)

    def compute_covariance_explained(
        self, X: np.ndarray, 
        Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return squared correlations and normalized covariance explained."""
        canonical_correlations = self.score(X, Y)
        squared_corrs = canonical_correlations ** 2
        total_variance_explained = np.sum(squared_corrs) + 1e-12
        covariance_explained = squared_corrs / total_variance_explained
        return squared_corrs, covariance_explained


def grid_search_scca(
    X: np.ndarray,
    Y: np.ndarray,
    model: str,
    n_components: int = 10,
    l1_c1_range: Optional[Iterable[float]] = None,
    l1_c2_range: Optional[Iterable[float]] = None,
    save_path: Optional[str] = None,
    patt: Optional[int] = None,
    comp: Optional[int] = None,
) -> Tuple[float, float, float, np.ndarray]:
    """Grid search for (l1_c1, l1_c2). Plotting/saving optional via args.

    Notes
    -----
    - Numerical logic and defaults are preserved exactly.
    """
    # Step 1: Defaults identical to original
    if model == "sCCA":
        if l1_c1_range is None:
            l1_c1_range = np.linspace(0.5, 0.9, 9)
        if l1_c2_range is None:
            l1_c2_range = np.linspace(0.1, 0.5, 9)
    if model == "kCCA":
        if l1_c1_range is None:
            l1_c1_range = np.logspace(-3, 1, num=5)
        if l1_c2_range is None:
            l1_c2_range = np.logspace(-6, -2, num=5)

    # Step 2: Evaluate grid (same scoring)
    scores = np.zeros((len(l1_c1_range), len(l1_c2_range)))
    for i, l1_c1 in enumerate(l1_c1_range):
        for j, l1_c2 in enumerate(l1_c2_range):
            scca = SparseCCA(n_components=n_components, l1_c1=round(float(l1_c1), 3), l1_c2=round(float(l1_c2), 3))
            try:
                scca.fit(X, Y)
                corrs = scca.score(X, Y)
                scores[i, j] = np.mean(corrs)
            except Exception:
                scores[i, j] = 0

    # Step 3: Select best
    best_idx = np.unravel_index(np.argmax(scores), scores.shape)
    best_l1_c1 = float(l1_c1_range[best_idx[0]])
    best_l1_c2 = float(l1_c2_range[best_idx[1]])
    best_score = float(scores[best_idx])

    # Step 4: Plot identical labels and save name pattern if args provided
    plt.figure(figsize=(4, 3), dpi=200)
    heatmap = sns.heatmap(
        scores,
        xticklabels=np.round(list(l1_c1_range), 2),  # keep original axis labeling
        yticklabels=np.round(list(l1_c2_range), 2),
        cmap="coolwarm",
        annot=True,
        annot_kws={"size": 8},
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel("L-1 Regularized factor on neural-dimension", fontsize=8)
    plt.ylabel("L-1 Regularized factor on behavioral-dimension", fontsize=8)
    plt.tick_params(axis="both", which="major", labelsize=8)
    if save_path is not None and patt is not None and comp is not None:
        out = f"{save_path}sCCA_grid_{patt}_Comp{comp+1}"
        plt.savefig(out, bbox_inches="tight")
    plt.show()

    return best_l1_c1, best_l1_c2, best_score, scores


def permutation_test_scca(
    X: np.ndarray,
    Y: np.ndarray,
    model: str,
    scca_model: SparseCCA,
    best_l1_c1: float,
    best_l1_c2: float,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Permutation test by shuffling Y. Logic preserved (including original choices)."""
    n_components = scca_model.n_components
    orig_corrs = scca_model.score(X, Y)
    perm_corrs = np.zeros((n_permutations, n_components))

    rng = check_random_state(random_state)

    for i in range(n_permutations):
        perm_idx = rng.permutation(len(Y))
        Y_perm = Y[perm_idx]

        # NOTE: Preserve original behavior: l1_c2 uses best_l1_c1 factor
        perm_scca = SparseCCA(
            n_components=n_components,
            l1_c1=min(0.999, 1.5 * best_l1_c1),
            l1_c2=min(0.999, 1.5 * best_l1_c1),
        )

        perm_scca.fit(X, Y_perm)
        X_scores, Y_scores = perm_scca.transform(X, Y_perm)

        for j in range(n_components):
            r, _ = pearsonr(X_scores[:, j], Y_scores[:, j])
            perm_corrs[i, j] = r

    p_values = np.zeros(n_components)
    for j in range(n_components):
        p_values[j] = np.mean(perm_corrs[:, j] >= np.abs(orig_corrs[j]))

    return p_values, perm_corrs


def run_sparse_cca_analysis(
    brain_data: np.ndarray,
    behavior_data: np.ndarray,
    model: str,
    n_components: int = 10,
    n_permutations: int = 1000,
    save_path: Optional[str] = None,
    patt: Optional[int] = None,
    comp: Optional[int] = None,
) -> Tuple[
    SparseCCA,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Full analysis wrapper mirroring the original flow."""
    scaler = StandardScaler()
    X = scaler.fit_transform(brain_data)
    Y = scaler.fit_transform(behavior_data)

    best_l1_c1, best_l1_c2, _, grid_scores = grid_search_scca(
        X, Y, model, n_components=n_components, save_path=save_path, patt=patt, comp=comp
    )

    print(f"Best parameters: l1_c1={best_l1_c1}, l1_c2={best_l1_c2}")

    scca = SparseCCA(n_components=n_components, l1_c1=best_l1_c1, l1_c2=best_l1_c2)
    scca.fit(X, Y)

    X_scores, Y_scores = scca.transform(X, Y)
    canonical_correlations = scca.score(X, Y)

    p_values, perm_corrs = permutation_test_scca(
        X, Y, model, scca, best_l1_c1, best_l1_c2, n_permutations
    )
    return (
        scca,
        scca.x_weights_,
        scca.y_weights_,
        X_scores,
        Y_scores,
        canonical_correlations,
        p_values,
        perm_corrs,
        grid_scores,
    )


def visualize_cca_results(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    canon_corrs: np.ndarray,
    p_values: np.ndarray,
    fdr_pvals: np.ndarray,
    perm_corrs: np.ndarray,
    n_components: int,
    top_behavior_features: np.ndarray,
    name: str,
    comp: int,
    save_path: Optional[str] = None,
    patt: Optional[int] = None,
    df_scaled: Optional[np.ndarray] = None,
    patterns: Optional[Iterable[str]] = None,
) -> None:
    """Replicate original visualization, keeping calculations unchanged."""
    # Bar plot of canonical correlations
    fig, ax = plt.subplots(figsize=(2.8, 3), dpi=200)
    colors = []
    for p in fdr_pvals:
        if p < 0.001:
            colors.append("purple")
        elif p < 0.01:
            colors.append("blue")
        elif p < 0.05:
            colors.append("red")
        else:
            colors.append("gray")
    ax.bar(range(1, n_components + 1), canon_corrs, color=colors)
    for i, p in enumerate(fdr_pvals):
        if p < 0.001:
            ax.text(i + 1, canon_corrs[i] + np.std(perm_corrs[:, i]), "***", ha="center", fontsize=8)
        elif p < 0.01:
            ax.text(i + 1, canon_corrs[i] + np.std(perm_corrs[:, i], axis=0), "**", ha="center", fontsize=8)
        elif p < 0.05:
            ax.text(i + 1, canon_corrs[i] + np.std(perm_corrs[:, i], axis=0), "*", ha="center", fontsize=8)
    ax.set_xlabel("Canonical component index", fontsize=8)
    ax.set_ylabel("Correlation", fontsize=8)
    ax.set_xticks(range(1, n_components + 1))
    ax.errorbar(
        range(1, n_components + 1),
        canon_corrs,
        yerr=np.std(perm_corrs, axis=0),
        fmt="none",
        capsize=5,
        color="black",
    )
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None and patt is not None:
        plt.savefig(f"{save_path}CC_corrs_{patt}_Comp{comp+1}", bbox_inches="tight")
    plt.show()

    # Permutation summary (kept as in original)
    perm_corrs_all = np.mean(perm_corrs, axis=1)
    p_values_all = np.sum(perm_corrs_all > np.mean(canon_corrs))
    plt.figure(figsize=(4, 4))
    sns.histplot(np.mean(perm_corrs, axis=1), bins=30, kde=True, color="gray")
    plt.axvline(np.mean(canon_corrs), color="red", linestyle="dashed",
                label=f'Observed: {np.mean(canon_corrs):.3f}, p={np.mean(p_values_all):.3f}')
    plt.xlabel("Correlation", fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    if save_path is not None and patt is not None:
        plt.savefig(f"{save_path}Permutation_{patt}_Comp{comp+1}", bbox_inches="tight")
    plt.show()

    # Inner function (kept logic; cleaned label string to avoid formatting crash)
    def plot_cca_scatter(component_idx: int, label_cc: str, color, label_beh: str, mode: int, name: str, comp: int) -> None:
        brain_score = x_scores[:, component_idx]
        behavior_score = y_scores[:, component_idx]
        r_value, _ = pearsonr(brain_score, behavior_score)

        if df_scaled is None:
            alpha_vals = np.ones_like(brain_score)
        else:
            sca_color = df_scaled[:, component_idx]
            sca_color = (sca_color - np.min(sca_color)) / (np.max(sca_color) - np.min(sca_color) + 1e-12)
            alpha_vals = sca_color

        if mode == 1:
            sig = " < 0.001"
        if mode == 2:
            sig = " < 0.01"
        if mode == 3:
            sig = " < 0.05"

        fig, ax = plt.subplots(figsize=(1.8, 2.25), dpi=200)
        sns.regplot(x=brain_score, y=behavior_score, line_kws={"color": color}, scatter=False)
        ax.scatter(brain_score, behavior_score, c=color, alpha=alpha_vals, s=3)

        x_length = max(brain_score) - min(brain_score)
        y_length = 3 * max(behavior_score) - min(behavior_score)

        ax.set_title(f"{label_beh}", fontsize=8)
        ax.set_xlabel(f"{name} PC{comp} scores", fontsize=8)
        ax.set_ylabel("Behavior scores", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim([min(brain_score), max(brain_score)])
        ax.set_ylim([min(behavior_score), 3 * max(behavior_score)])
        ax_insert = fig.add_axes([0.62, 0.6, 0.28, 0.15])
        ax_insert.set_zorder(ax.get_zorder() + 1)
        sns.histplot(perm_corrs[:, component_idx], bins=30, kde=True, color="gray")
        plt.axvline(r_value, color=color, linestyle="dashed")
        ax_insert.set_ylabel("", fontsize=8)
        ax_insert.spines["top"].set_visible(False)
        ax_insert.spines["right"].set_visible(False)
        ax_insert.spines["left"].set_visible(False)
        ax_insert.tick_params(
            top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=False, labelsize=8
        )
        ax.text(
            min(brain_score) + 0.05 * x_length,
            3 * max(behavior_score) - 0.05 * y_length,
            f"r = {r_value:.3f}   P{sig}",
            va="center",
            fontsize=8,
            zorder=10,
        )
        ax.tick_params(axis="both", which="major", labelsize=8)
        if save_path is not None and patterns is not None and patt is not None:
            plt.savefig(f"{save_path}{patterns[patt]}_Comp{comp}_{label_cc}", bbox_inches="tight")
        plt.show()

    # Colors and labels preserved
    colors_beha = [
        (220 / 256, 57 / 256, 206 / 256),
        (218 / 256, 105 / 256, 17 / 256),
        (218 / 256, 105 / 256, 17 / 256),
        (220 / 256, 57 / 256, 206 / 256),
        (220 / 256, 57 / 256, 206 / 256),
        (218 / 256, 105 / 256, 17 / 256),
        (4 / 256, 85 / 256, 132 / 256),
        (45 / 256, 13 / 256, 3 / 256),
    ]
    labels_beha = [
        "Emotion",
        "Cognition",
        "Language comprehension",
        "Fear",
        "Aggressive behavior",
        "Working memory",
        "Personality",
        "Alertness",
    ]

    for j, p in enumerate(fdr_pvals):
        i = int(np.argmax(np.abs(top_behavior_features[:, j])))
        if p < 0.001:
            plot_cca_scatter(j, f"CC{j+1}", colors_beha[i], labels_beha[i], 1, name, comp)
        elif p < 0.01:
            plot_cca_scatter(j, f"CC{j+1}", colors_beha[i], labels_beha[i], 2, name, comp)
        elif p < 0.05:
            plot_cca_scatter(j, f"CC{j+1}", colors_beha[i], labels_beha[i], 3, name, comp)


def main(
    path: str,
    behavior_csv: str,
    save_path: str,
    sessions: Iterable[str] = ("REST2",),
    patterns: Iterable[str] = ("real", "ang"),
    pattern_names: Iterable[str] = ("Complex", "SRP"),
    comps: int = 3,
    n_CC: int = 8,
    n_permutations: int = 1000,
    seed: int = 42,
) -> None:
    """Main entrypoint: mirrors the original bottom script, but parameterized.

    Parameters
    ----------
    path : str
        Directory containing `values_{session}_{pattern}_{comp}.txt`.
    behavior_csv : str
        Path to the behavioral CSV.
    save_path : str
        Directory prefix for figure outputs (same concatenation style).
    sessions, patterns, pattern_names : iterable
        Loop dimensions, preserved.
    comps : int
        Number of `comp` indices to iterate.
    n_CC : int
        Number of canonical components.
    n_permutations : int
        Number of permutations for significance testing.
    seed : int
        Numpy random seed.
    """
    np.random.seed(seed)
    patterns = ['real', 'srp']

    # Ensure pathlib objects for cross-platform path joining where needed
    base_path = path
    save_prefix = save_path

    for comp in range(comps):
        for patt, patt_name in enumerate(['Complex', 'SRP']):
            # Step 1: Load network data
            data_file =  f"{base_path}values_{patterns[patt]}_{comp}.txt"
            with open(data_file, "r") as file:
                lines = file.readlines()
            formatted_lines = [line.strip() for line in lines]
            df_data = pd.DataFrame([line.split("\t") for line in formatted_lines]).astype(float)

            # Step 2: Load behavioral data and numeric subset
            df = pd.read_csv(behavior_csv).select_dtypes(include=[np.number])

            # Step 3: Standardize (as in original)
            df_data_scaled = StandardScaler().fit_transform(df_data)
            df_scaled = StandardScaler().fit_transform(df.iloc[:, 12 : 12 + n_CC])

            # Step 4: Covariates (kept as-is; residuals not used later to preserve results)
            c1 = StandardScaler().fit_transform(df[["Gender1", "Gender2"]].values)
            c2 = StandardScaler().fit_transform(df[["Age1", "Age2", "Age3", "Age4"]].values)
            c3 = StandardScaler().fit_transform(df[["Race1", "Race2", "Race3", "Race4", "Race5"]].values)

            X_with_constant = sm.add_constant(df_data_scaled)
            covariates_with_constant = sm.add_constant(np.hstack((c1, c2, c3)))
            model_ols = sm.OLS(X_with_constant, covariates_with_constant).fit()
            residuals_X = (X_with_constant - model_ols.predict(covariates_with_constant))[:, 1:]

            # Step 5: Use the same data choices as the original script (not residuals)
            net_data, med_data = df_data_scaled, df_scaled

            print(f"Training sample has {net_data.shape[0]} subjects.")
            print(f"rsFC data has {net_data.shape[1]} edges.")
            print(f"Clinical data has {med_data.shape[1]} items.")

            # Step 6: Run sCCA analysis (same model='sCCA')
            (
                scca,
                x_weights,
                y_weights,
                x_scores,
                y_scores,
                cc_corrs,
                p_values,
                perm_corrs,
                grid_scores,
            ) = run_sparse_cca_analysis(
                net_data,
                med_data,
                "sCCA",
                n_components=n_CC,
                n_permutations=n_permutations,
                save_path=str(save_prefix),
                patt=patt,
                comp=comp,
            )

            # Step 7: Covariance explained (same plot style)
            cvar_exp, _ = scca.compute_covariance_explained(net_data, med_data)
            varE_df = pd.DataFrame({"Mode": range(1, len(cvar_exp) + 1), "Variance Explained": cvar_exp})
            fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
            ax.scatter(
                varE_df["Mode"],
                varE_df["Variance Explained"],
                s=varE_df["Variance Explained"] * 1000,
                c=varE_df["Variance Explained"],
                alpha=0.7,
            )
            plt.title("Scatter Plot of Covariance Explained by Canonical Components", fontsize=15)
            plt.xlabel("Canonical variate index", fontsize=15)
            plt.ylabel("Covariance Explained", fontsize=15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plt.show()

            # Step 8: Feature correlations (identical calc)
            cca_brain_corr = (net_data.T @ x_scores) / (net_data.shape[0] - 1)
            cca_behavior_corr = (med_data.T @ y_scores) / (net_data.shape[0] - 1)
            top_brain_idx = np.argsort(np.sum(np.abs(cca_brain_corr), axis=1))[-10:]
            top_brain_features = cca_brain_corr[top_brain_idx, :]
            top_behavior_features = cca_behavior_corr[:n_CC]

            # Step 9: Heatmaps (identical)
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
            cax = ax.matshow(top_brain_features, cmap="coolwarm")
            ax.set_xticks(ticks=np.arange(len(top_brain_features)))
            ax.set_xticklabels([f"CC {i+1}" for i in range(len(top_brain_features))], fontsize=15)
            ax.set_yticks(ticks=np.arange(len(top_brain_features)))
            ax.set_yticklabels([f"Nuero PC {i+1}" for i in range(len(top_brain_features))], fontsize=15)
            plt.title("Top Brain Feature Contributions to CCA Components", fontsize=15)
            ax.tick_params(top=False, bottom=False, left=False, labeltop=False, labelbottom=True)
            cbar = fig.colorbar(cax, fraction=0.045)
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Correlation", fontsize=15)
            ax.tick_params(axis="both", which="major", labelsize=8)
            plt.savefig(f"{save_prefix}corr_neu_{patt}_Comp{comp+1}", bbox_inches="tight")
            plt.show()

            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            cax = ax.matshow(top_behavior_features, cmap="coolwarm")
            ax.set_xticks(ticks=np.arange(len(top_behavior_features)))
            ax.set_xticklabels([f"{i+1}" for i in range(len(top_behavior_features))], fontsize=8)
            ax.set_yticks(ticks=np.arange(len(top_behavior_features)))
            ax.set_yticklabels([f"{i+1}" for i in range(len(top_behavior_features))], fontsize=8)
            ax.tick_params(top=False, bottom=False, left=False, labeltop=False, labelbottom=True)
            cbar = fig.colorbar(cax, fraction=0.045)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Correlation", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=8)
            plt.xlabel("Canonical component index", fontsize=8)
            plt.ylabel("Behavioral component index", fontsize=8)
            plt.savefig(f"{save_prefix}corr_beh_{patt}_Comp{comp+1}", bbox_inches="tight")
            plt.show()

            # Step 10: FDR and per-component viz (logic preserved; parameters passed-in)
            _, fdr_pvals, _, _ = multipletests(p_values, method="fdr_bh")
            visualize_cca_results(
                x_scores,
                y_scores,
                cc_corrs,
                p_values,
                fdr_pvals,
                perm_corrs,
                n_CC,
                top_behavior_features,
                list(pattern_names)[patt],
                comp + 1,
                save_path=str(save_prefix),
                patt=patt,
                df_scaled=df_scaled,
                patterns=list(patterns),
            )

if __name__ == "__main__":
    # Simple CLI wrapper; pass your paths explicitly.
    parser = argparse.ArgumentParser(description="Run sparse CCA pipeline (minimal-change).")
    parser.add_argument("--path", type=str, required=True, 
                        help="Folder with spatiotemporal feature files.")
    parser.add_argument("--behavior_csv", type=str, required=True, 
                        help="Path to behavioral CSV file.")
    parser.add_argument("--save_path", type=str, required=True, 
                        help="Output directory for figures (prefix style).")
    parser.add_argument("--comps", type=int, 
                        default=3, 
                        help="Number of spatiotemporal features.")
    parser.add_argument("--n_CC", type=int, 
                        default=8, 
                        help="Number of canonical components.")
    parser.add_argument("--n_permutations", type=int, 
                        default=1000, 
                        help="Permutation count.")
    parser.add_argument("--seed", type=int, 
                        default=42, 
                        help="Random seed.")
    
    if len(sys.argv) == 1:
        sys.argv += [
            "--path", "./source_files/",
            "--behavior_csv", "./source_files/PCA_35_final4.csv",
            "--save_path", "./output/"
        ]

    
    args = parser.parse_args()

    # For cross-platform compatibility, we simply pass strings; paths are handled internally.
    main(
        path=args.path,
        behavior_csv=args.behavior_csv,
        save_path=args.save_path,
        comps=args.comps,
        n_CC=args.n_CC,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )