import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os
import torch
import gc

# =====================================================
# Global params
# =====================================================
SIM_LENGTH = 1728.0
TR = 0.72
DT_NEURAL = 0.01
TIME_SCALE = 2400
N_CPCA_COMPONENTS = 10

BASE_RESULTS_DIR = "sim_results_balloon_grid"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Balloon model (完全不改)
# =====================================================
class BOLD:
    def __init__(self, epsilon, tao_s, tao_f, tao_0, alpha, E_0, V_0, delta_t):
        self.epsilon = epsilon
        self.tao_s = tao_s
        self.tao_f = tao_f
        self.tao_0 = tao_0
        self.E_0 = E_0
        self.V_0 = V_0
        self.delta_t = delta_t
        self.div_alpha = 1.0 / alpha

        self.s = None
        self.q = None
        self.v = None
        self.f_in = None

    def update(self, name, df):
        x = getattr(self, name)
        if x is None:
            setattr(self, name, df * self.delta_t)
        else:
            x += df * self.delta_t

    def run(self, u):
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u.astype(np.float32)).to(DEVICE)

        if self.s is None:
            self.s = torch.zeros_like(u)
        if self.q is None:
            self.q = torch.ones_like(u)
        if self.v is None:
            self.v = torch.ones_like(u)
        if self.f_in is None:
            self.f_in = torch.ones_like(u)

        d_s = self.epsilon * u - self.s / self.tao_s - (self.f_in - 1.0) / self.tao_f

        q_part = torch.where(
            self.f_in > 0,
            1 - (1 - self.E_0) ** (1 / self.f_in),
            torch.ones_like(self.f_in),
        )

        self.update(
            "q",
            (self.f_in * q_part / self.E_0 - self.q * self.v ** (self.div_alpha - 1)) / self.tao_0
        )
        self.update(
            "v",
            (self.f_in - self.v ** self.div_alpha) / self.tao_0
        )
        self.update("f_in", self.s)
        self.f_in = torch.clamp(self.f_in, min=1e-5)
        self.update("s", d_s)

        bold = self.V_0 * (
            7 * self.E_0 * (1 - self.q)
            + 2 * (1 - self.q / self.v)
            + (2 * self.E_0 - 0.2) * (1 - self.v)
        )
        return bold

# =====================================================
# Wilson–Cowan neural field（关键修改在这里）
# =====================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def simulate_wilson_cowan_field(
    grid_shape, sim_length, dt,
    # 下面两个参数仍然保留，便于你 grid search
    sigma_noise=0.06,
    # 多时间尺度：覆盖 1–100s（你也可以再加更慢的 200s）
    tau_set=(1.0, 3.0, 10.0, 30.0, 100.0),
    # 每个时间尺度的权重（和为 1 更直观）
    tau_weights=(0.10, 0.15, 0.25, 0.25, 0.25),
    # 噪声空间相关程度
    noise_spatial_sigma=1.0,
):
    """
    Wilson–Cowan + 多时间尺度 OU 噪声（近似 1/f）
    - 不改 BOLD 类
    - 通过多个 OU time constants 的叠加产生更稳定的幂律段
    """
    nx, ny = grid_shape
    n_steps = int(sim_length / dt)

    E = np.random.randn(nx, ny)
    I = np.random.randn(nx, ny)

    tau_set = np.array(tau_set, dtype=np.float64)
    w = np.array(tau_weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    # 为每个 tau 维护一个 OU 状态
    etaE = [np.zeros((nx, ny), dtype=np.float64) for _ in tau_set]
    etaI = [np.zeros((nx, ny), dtype=np.float64) for _ in tau_set]

    E_history = np.zeros((n_steps, nx, ny), dtype=np.float32)

    for t in range(n_steps):
        E_c = gaussian_filter(E, sigma=0.6)
        I_c = gaussian_filter(I, sigma=0.6)

        # 叠加多尺度 OU
        noise_E = np.zeros((nx, ny), dtype=np.float64)
        noise_I = np.zeros((nx, ny), dtype=np.float64)

        for k, tau in enumerate(tau_set):
            # OU 更新：eta += dt*(-eta/tau) + sqrt(dt)*N(0,1)
            etaE[k] += dt * (-etaE[k] / tau) + np.sqrt(dt) * np.random.randn(nx, ny)
            etaI[k] += dt * (-etaI[k] / tau) + np.sqrt(dt) * np.random.randn(nx, ny)

            # 空间相关：对每个分量做轻微平滑
            e_k = gaussian_filter(etaE[k], sigma=noise_spatial_sigma)
            i_k = gaussian_filter(etaI[k], sigma=noise_spatial_sigma)

            noise_E += w[k] * e_k
            noise_I += w[k] * i_k

        dE = (-E + sigmoid(10.5 * E_c - 9.8 * I_c + 0.2)) / 0.08 + sigma_noise * noise_E
        dI = (-I + sigmoid(10.2 * E_c - 9.7 * I_c + 0.1)) / 0.12 + sigma_noise * noise_I

        E += dt * dE
        I += dt * dI

        E_history[t] = E.astype(np.float32)

    return E_history


# =====================================================
# Load cerebellum voxels
# =====================================================
def load_cerebellum_voxels():
    work_path = "D:/Desktop/2025_CPCA_Cerebellum/Code-Three"
    cifti_path = f"{work_path}/Yeo7_Buckner-Choi-Raut-combined.dtseries.nii"
    mni_template_path = f"{work_path}/FSL_MNI152_2mm.nii.gz"

    cifti = nib.load(cifti_path)
    bm = cifti.header.get_index_map(1).brain_models
    affine = nib.load(mni_template_path).affine

    coords = []
    cere_idx = []
    i_map = 0
    for model in bm:
        i_map += 1
        if model.model_type == "CIFTI_MODEL_TYPE_VOXELS" and i_map in (10, 11):
            xyz = apply_affine(affine, model.voxel_indices_ijk)
            coords.append(xyz)
            cere_idx.extend(
                np.arange(model.index_offset, model.index_offset + model.index_count)
            )

    return np.vstack(coords), np.array(cere_idx), cifti.header

def run_cpca(ts_delayed, n_components=N_CPCA_COMPONENTS):
    """
    对 voxel-level BOLD 做 Hilbert transform，构造 analytic signal，
    然后在 N×T 矩阵上做 SVD (或 randomized SVD) 得到复主成分。

    输入:
        ts_delayed : (N_vox, T)
        n_components : 提取的组件数

    返回:
        U : (N_vox, n_components) 复空间主成分
        S : (n_components,)
        Vh: (n_components, T)
    """
    from sklearn.utils.extmath import randomized_svd

    analytic = hilbert(ts_delayed, axis=1)
    X = (analytic / np.abs(analytic + 1e-8)).astype(np.complex64)

    U, S, Vh = randomized_svd(X, n_components=n_components)

    return U, S, Vh

def add_traveling_wave_delay(voxel_ts, voxel_xyz, max_delay_tr=5):
    """
    在 voxel-level BOLD 上加入空间相关的 time-lag 结构。
    举例: delay 随 y 坐标增加，从 0 ~ max_delay_samples 个 TR。

    输入:
        voxel_bold : (N_vox, T)
        voxel_xyz  : (N_vox, 3)
        max_delay_samples : 最大延迟 (单位: BOLD 采样点数)

    返回:
        ts_delayed : (N_vox, T) 加了滞后之后的 BOLD
        delay_map  : (N_vox,) 每个 voxel 的样本延迟
    """
    y = voxel_xyz[:, 1]
    d = (y - y.min()) / (y.max() - y.min() + 1e-6)
    d = (d * max_delay_tr).astype(int)

    out = np.zeros_like(voxel_ts)
    for i in range(voxel_ts.shape[0]):
        out[i] = np.roll(voxel_ts[i], d[i])

    return out, d

# =====================================================
# Main (GRID SEARCH on neural noise)
# =====================================================
voxel_xyz, cere_idx, hdr = load_cerebellum_voxels()
voxel_xy = PCA(n_components=2).fit_transform(voxel_xyz)

nx = ny = 100

# === 正确的 grid：神经噪声参数 ===
tau_noise_list   = [10.0, 20.0, 50.0]   # 秒（关键）
sigma_noise_list = [0.03, 0.06, 0.1]    # 小幅慢噪声
tau_f_list = [1.0, 1.5, 2.0]

grid_results = []

for tau_noise in tau_noise_list:
    for sigma_noise in sigma_noise_list:

        print(f"\nNeural noise: tau={tau_noise}, sigma={sigma_noise}")

        # 神经场只跑一次
        E_hist = simulate_wilson_cowan_field(
            (nx, ny), SIM_LENGTH, DT_NEURAL,
            sigma_noise=sigma_noise,
            tau_set=(1.0, 3.0, 10.0, 30.0, 100.0),
            tau_weights=(0.10, 0.15, 0.25, 0.25, 0.25),
            noise_spatial_sigma=1.0,
        )

        T_neural = E_hist.shape[0]
        u = E_hist.reshape(T_neural, -1)
        u -= u.mean(axis=0, keepdims=True)
        u = np.maximum(u, 0.0)
        u /= (u.std(axis=0, keepdims=True) + 1e-6)

        for tau_f in tau_f_list:

            tag = f"tn{tau_noise}_sn{sigma_noise}_tf{tau_f}"
            print("Running:", tag)

            RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, tag)
            os.makedirs(RESULTS_DIR, exist_ok=True)

            bold_model = BOLD(
                epsilon=0.4,
                tao_s=1.0,
                tao_f=tau_f,
                tao_0=1.0,
                alpha=0.32,
                E_0=0.34,
                V_0=0.02,
                delta_t=DT_NEURAL
            )

            bold_cont = np.zeros_like(u, dtype=np.float32)
            for t in range(T_neural):
                bold_cont[t] = bold_model.run(u[t]).detach().cpu().numpy()

            step_per_tr = int(TR / DT_NEURAL)
            bold_tr = bold_cont[::step_per_tr][:TIME_SCALE]

            # ---- grid → voxel ----
            grid_x, grid_y = np.meshgrid(
                np.linspace(voxel_xy[:, 0].min(), voxel_xy[:, 0].max(), nx),
                np.linspace(voxel_xy[:, 1].min(), voxel_xy[:, 1].max(), ny)
            )
            points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

            voxel_ts = np.zeros((voxel_xyz.shape[0], bold_tr.shape[0]), dtype=np.float32)
            for t in range(bold_tr.shape[0]):
                voxel_ts[:, t] = griddata(
                    points,
                    bold_tr[t].reshape(nx, ny).ravel(),
                    voxel_xy,
                    method="nearest"
                )

            voxel_ts -= voxel_ts.mean(axis=1, keepdims=True)
            
            bold_noise_sigma = 0.01   # <<< 可调：0.005 ~ 0.02 都合理
            voxel_ts_noisy = voxel_ts + bold_noise_sigma * np.random.randn(
                *voxel_ts.shape
            ).astype(np.float32)

            # ---- PSD ----
            fs = 1 / TR
            idx = np.random.choice(voxel_ts.shape[0], 200, replace=False)
            f, p = welch(voxel_ts[idx], fs, axis=1, nperseg=512)
            psd_mean = p.mean(axis=0)

            mask = (f >= 0.02) & (f <= 0.08)
            beta = -np.polyfit(np.log(f[mask]), np.log(psd_mean[mask]), 1)[0]

            # ---- save ----
            np.save(os.path.join(RESULTS_DIR, "freqs.npy"), f)
            np.save(os.path.join(RESULTS_DIR, "psd_mean.npy"), psd_mean)
            np.save(os.path.join(RESULTS_DIR, "voxel_ts.npy"), voxel_ts)
            
            ts_delayed, delay_map = add_traveling_wave_delay(
                voxel_ts, 
                voxel_xyz, 
                max_delay_tr=2*np.pi
            )
            
            U, S, Vh = run_cpca(ts_delayed, n_components=N_CPCA_COMPONENTS)
            def compute_explained_variance(S):
                """
                S 是奇异值（复 PCA 返回的）
                explained variance = S^2 / sum(S^2)
                """
                var = S**2
                exp_var = var / np.sum(var)
                cum_var = np.cumsum(exp_var)
                return exp_var, cum_var
            
            exp_var, cum_var = compute_explained_variance(S)
            
            print("Explained variance:", exp_var)
            print("Cumulative explained variance:", cum_var)
            
            phase = np.angle(U[:, :])
            for comp in range(N_CPCA_COMPONENTS):
                print("Delay correlation:", np.corrcoef(phase[:, comp], delay_map)[0,1])
            
            hdr_axis0  = hdr.get_axis(0)
            hdr_axis0.size = N_CPCA_COMPONENTS + 1
            hdr_axis1 = hdr.get_axis(1)
            phase_all = np.full((hdr_axis1.size, hdr_axis0.size), np.nan, dtype=np.float32)
            phase_all[cere_idx, 0] = delay_map.astype(np.float32)
            phase_all[cere_idx, 1:1+N_CPCA_COMPONENTS] = phase.astype(np.float32)
            cifti_out = nib.Cifti2Image(phase_all.T, (hdr_axis0, hdr_axis1))
            nib.save(cifti_out, f'Simulated_phase_{N_CPCA_COMPONENTS}_results_{tag}.dtseries.nii')
            
            
            
            

            grid_results.append({
                "tau_noise": tau_noise,
                "sigma_noise": sigma_noise,
                "tau_f": tau_f,
                "beta": beta
            })

            # ---- plot (log–log 才能看 1/f) ----
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            ax.loglog(f[1:], psd_mean[1:], linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.set_title("Simulated BOLD PSD (Balloon)")
            plt.show()
            
            vid = np.random.choice(idx)
            f1, p1 = welch(
                voxel_ts_noisy[vid],
                fs,
                nperseg=512,
                detrend="constant"
            )
            
            plt.figure(figsize=(4,4), dpi=300)
            plt.loglog(f1[1:], p1[1:], linewidth=1)
            plt.xlim(0.01, 1)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")
            plt.title("Single-voxel BOLD PSD (noisy)")
            plt.show()

            print("  beta =", round(beta, 3))

            del bold_cont, bold_tr, voxel_ts
            gc.collect()

# ---- save summary ----
np.save(os.path.join(BASE_RESULTS_DIR, "grid_results.npy"), grid_results)
print("GRID SEARCH DONE.")
