"""
Netflix Prize - DP Matrix Factorization Experiments
=====================================================
Run this AFTER movielens and anime are done.
Produces: fig1, fig4, fig7, fig10, fig13, fig16, fig19, fig22, fig25 for Netflix.

Netflix is too large for dense GPU mode (2GB+ matrix).
This script forces CHUNKED mode and limits n-sweep to n <= 100
to avoid CUDA OOM on 8GB VRAM.

Install (Anaconda Prompt):
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  conda install scipy pandas matplotlib
"""

import numpy as np
import pandas as pd
import os
import time
import threading
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.sparse import csr_matrix

# =============================================================================
# DEVICE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 55)
print(f"  Device : {device}")
if device.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("=" * 55)
print()

BASE       = "D:/College/YEAR 3/SEM 6/Recommendation System/project"
OUTPUT_DIR = os.path.join(BASE, "outputs")
# Netflix is large — process 100 rows at a time to stay in VRAM
CHUNK      = 100


# =============================================================================
# PAUSE / RESUME CONTROLLER
# =============================================================================

class PauseController:
    def __init__(self):
        self.paused    = False
        self.stop_flag = False
        self._thread   = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print("  Controls: P=Pause  R=Resume  S=Save&Stop  (type + Enter)")

    def _listen(self):
        while True:
            try:
                ch = input().strip().upper()
                if ch == "P" and not self.paused:
                    self.paused = True
                    print("\n  [PAUSED]  R=resume  S=save&stop")
                elif ch == "R" and self.paused:
                    self.paused = False
                    print("  [RESUMED]")
                elif ch == "S":
                    self.stop_flag = True
                    self.paused    = False
                    print("  [STOPPING after this iteration...]")
            except EOFError:
                break

    def check(self):
        while self.paused:
            time.sleep(0.3)
        return self.stop_flag


_pause_ctrl = None

def start_pause_controller():
    global _pause_ctrl
    _pause_ctrl = PauseController()

def get_pause_ctrl():
    return _pause_ctrl


# =============================================================================
# DATA LOADER - Netflix only
# =============================================================================

def load_netflix(folder):
    # Paper reports 5466 movies, 11345 users, 8.65% density (~5.36M ratings)
    # This matches combined_data_1.txt only (movies 1-4499, ~5M ratings)
    # Using all 4 files gives 53k users / 44M ratings which is too large
    files = ["combined_data_1.txt"]
    records = []
    for fname in files:
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            continue
        print(f"  Parsing {fname}...")
        current_movie = None
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    current_movie = int(line[:-1])
                elif current_movie:
                    p = line.split(",")
                    if len(p) >= 2:
                        records.append((current_movie, int(p[0]), float(p[1])))
    df = pd.DataFrame(records, columns=["movie", "user", "rating"])
    print(f"  Raw: {len(df):,} ratings")
    uc = df["user"].value_counts()
    mc = df["movie"].value_counts()
    df = df[df["user"].isin(uc[(uc >= 200) & (uc <= 2500)].index) &
            df["movie"].isin(mc[mc > 100].index)]
    print(f"  After filter: {len(df):,} ratings | "
          f"{df['user'].nunique():,} users | {df['movie'].nunique():,} movies")
    return df


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess(df):
    user_ids  = sorted(df["user"].unique())
    movie_ids = sorted(df["movie"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    idx2user  = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}

    df = df.copy()
    df["user"]  = df["user"].map(user2idx)
    df["movie"] = df["movie"].map(movie2idx)

    n_users  = len(user_ids)
    n_movies = len(movie_ids)
    density  = len(df) / (n_users * n_movies) * 100
    mem_gb   = n_movies * n_users * 4 / 1e9

    print(f"  Users   : {n_users:,}  |  Movies : {n_movies:,}")
    print(f"  Ratings : {len(df):,}  (density {density:.2f}%)")
    print(f"  Dense   : {mem_gb:.2f} GB  ->  FORCED chunked=True")

    rows = df["movie"].values
    cols = df["user"].values
    vals = df["rating"].values.astype(np.float32)
    ones = np.ones(len(df), dtype=np.float32)

    V_sp = csr_matrix((vals, (rows, cols)), shape=(n_movies, n_users))
    R_sp = csr_matrix((ones, (rows, cols)), shape=(n_movies, n_users))

    return V_sp, R_sp, n_movies, n_users, user2idx, movie2idx, idx2user, idx2movie, df


def train_test_split_sparse(V_sp, R_sp, test_ratio=0.1, seed=42):
    R_coo      = R_sp.tocoo()
    rows, cols = R_coo.row, R_coo.col
    rng        = np.random.default_rng(seed)
    idx        = np.arange(len(rows))
    rng.shuffle(idx)
    split  = int(len(idx) * test_ratio)
    ti, vi = idx[:split], idx[split:]
    sh     = R_sp.shape

    R_train = csr_matrix((np.ones(len(vi), dtype=np.float32),
                          (rows[vi], cols[vi])), shape=sh)
    R_test  = csr_matrix((np.ones(len(ti), dtype=np.float32),
                          (rows[ti], cols[ti])), shape=sh)

    V_coo  = V_sp.tocoo()
    V_dict = {(r, c): v for r, c, v in zip(V_coo.row, V_coo.col, V_coo.data)}
    test_v = np.array([V_dict.get((rows[i], cols[i]), 0.0) for i in ti],
                      dtype=np.float32)
    V_test = csr_matrix((test_v, (rows[ti], cols[ti])), shape=sh)

    print(f"  Train: {R_train.nnz:,}   Test: {R_test.nnz:,}")
    return R_train, R_test, V_test


# =============================================================================
# ALGORITHM 1
# =============================================================================

class DPMatrixFactorization:
    def __init__(self, n=20, eps_i=0.30, delta=0.01, delta_r=1e-5,
                 lam=0.01, mu=0.0005, C=1.0,
                 max_iter=2500, tol=1e-4, private=True, seed=42):
        self.n=n; self.eps_i=eps_i; self.delta=delta; self.delta_r=delta_r
        self.lam=lam; self.mu=mu; self.C=C
        self.max_iter=max_iter; self.tol=tol
        self.private=private; self.seed=seed
        self.X=None; self.Theta=None
        self.rmse_hist=[]; self.time_per_iter=[]; self.J=0

    def sigma(self, tau):
        return (tau * self.C / self.eps_i) * np.sqrt(2.0 * np.log(1.25 / self.delta))

    def overall_epsilon(self):
        J, ei, d, dr = self.J, self.eps_i, self.delta, self.delta_r
        denom = 4.0 * np.log(1.25 / d)
        return J * ei**2 / denom + 2.0 * np.sqrt(J * ei**2 * np.log(1.0 / dr) / denom)

    @staticmethod
    def _unit_norm(M):
        return M / torch.linalg.norm(M, dim=1, keepdim=True).clamp(min=1e-12)

    @staticmethod
    def _clip(M, C):
        return M / torch.clamp(torch.linalg.norm(M, dim=1, keepdim=True) / C, min=1.0)

    def _rmse_sparse(self, V_test_sp, R_test_sp, X, Theta):
        """Compute RMSE in small batches to avoid OOM on large test sets."""
        coo    = R_test_sp.tocoo()
        all_ri = coo.row
        all_ci = coo.col
        true_v = V_test_sp.tocoo().data.astype(np.float32)

        BATCH = 500_000
        sq_err_sum = 0.0
        n_total    = len(all_ri)

        for s in range(0, n_total, BATCH):
            e    = min(s + BATCH, n_total)
            ri   = torch.tensor(all_ri[s:e], dtype=torch.long,    device=device)
            ci   = torch.tensor(all_ci[s:e], dtype=torch.long,    device=device)
            true = torch.tensor(true_v[s:e], dtype=torch.float32, device=device)
            pred = (X[ri] * Theta[ci]).sum(dim=1)
            sq_err_sum += float(((pred - true) ** 2).sum().item())

        return float(np.sqrt(sq_err_sum / n_total))

    def fit(self, V_sp, R_train_sp, R_test_sp, V_test_sp,
            tau, n_movies, n_users, verbose=True, record_time=False):
        """Always uses chunked mode for Netflix."""
        torch.manual_seed(self.seed)
        sig   = float(self.sigma(tau)) if self.private else 0.0
        X     = self._unit_norm(torch.randn(n_movies, self.n, device=device))
        Theta = self._unit_norm(torch.randn(n_users,  self.n, device=device))

        prev_rmse = float("inf")
        t0        = time.time()

        for t in range(1, self.max_iter + 1):
            iter_t0 = time.time()

            # Chunked gradient computation
            gX  = self.lam * X.clone()
            gTh = self.lam * Theta.clone()
            for s in range(0, n_movies, CHUNK):
                e   = min(s + CHUNK, n_movies)
                Vc  = torch.tensor(V_sp[s:e].toarray(),
                                   dtype=torch.float32, device=device)
                Rc  = torch.tensor(R_train_sp[s:e].toarray(),
                                   dtype=torch.float32, device=device)
                Xc  = X[s:e]
                res = (Xc @ Theta.T - Vc) * Rc
                gX[s:e] += res @ self._clip(Theta, self.C)
                gTh     += res.T @ self._clip(Xc, self.C)
                del Vc, Rc, res

            if self.private:
                gTh += torch.randn_like(gTh) * sig

            X     = X     - self.mu * gX
            Theta = Theta - self.mu * gTh

            if record_time:
                self.time_per_iter.append((time.time() - iter_t0) * 1000)

            if t % 50 == 0 or t == 1:
                with torch.no_grad():
                    rmse = self._rmse_sparse(V_test_sp, R_test_sp, X, Theta)
                self.rmse_hist.append((t, rmse))

                tag = f"eps_i={self.eps_i}" if self.private else "non-private"
                if verbose and t % 500 == 0:
                    print(f"    iter {t:5d} [{tag}]  RMSE={rmse:.4f}  ({time.time()-t0:.0f}s)")

                if abs(prev_rmse - rmse) < self.tol and t > 200 and rmse < 10.0:
                    if verbose:
                        print(f"    Converged iter {t} [{tag}]  RMSE={rmse:.4f}  ({time.time()-t0:.0f}s)")
                    self.J = t
                    break
                prev_rmse = rmse

                ctrl = get_pause_ctrl()
                if ctrl is not None and ctrl.check():
                    print(f"    Stopped at iter {t}  RMSE={rmse:.4f}")
                    self.J     = t
                    self.X     = X.cpu().numpy()
                    self.Theta = Theta.cpu().numpy()
                    torch.cuda.empty_cache()
                    return self
        else:
            self.J = self.max_iter

        self.X     = X.cpu().numpy()
        self.Theta = Theta.cpu().numpy()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self


# =============================================================================
# EXPERIMENTS
# =============================================================================

def _fit(n, ei, mu, private, tau, V, Rt, Re, Vt, nm, nu,
         max_iter=2500, record_time=False):
    m = DPMatrixFactorization(n=n, eps_i=ei, mu=mu,
                               private=private, max_iter=max_iter)
    m.fit(V, Rt, Re, Vt, tau, nm, nu,
          verbose=False, record_time=record_time)
    return m


def save_model(m, path):
    with open(path, "wb") as f:
        pickle.dump({"X": m.X, "Theta": m.Theta,
                     "rmse_hist": m.rmse_hist, "J": m.J}, f)
    print(f"  Saved: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def run_netflix_experiments(V, Rt, Re, Vt, tau, nm, nu):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    label = "Netflix dataset"
    name  = "netflix"

    eps5    = [0.20, 0.30, 0.50, 0.75, 0.90]
    markers = ["^", "v", "s", "D", "*"]

    # =========================================================
    # Exp 1: RMSE vs Iterations  (Fig 1-3)
    # =========================================================
    print(f"\n{'='*50}")
    print(f"  Exp 1: RMSE vs Iterations  (Fig 1-3)")
    print(f"{'='*50}")
    res_iter = {}
    print("  [non-private]")
    m = _fit(20, 0.30, 0.0005, False, tau, V, Rt, Re, Vt, nm, nu, 2500)
    res_iter["non_private"] = m.rmse_hist
    save_model(m, os.path.join(OUTPUT_DIR, "netflix_iter_nonprivate.pkl"))

    for ei in eps5:
        print(f"  [eps_i={ei}]")
        m = _fit(20, ei, 0.0005, True, tau, V, Rt, Re, Vt, nm, nu, 2500)
        res_iter[ei] = m.rmse_hist
        save_model(m, os.path.join(OUTPUT_DIR, f"netflix_iter_eps{ei}.pkl"))

    plt.figure(figsize=(5.5, 4))
    iters, rmse = zip(*res_iter["non_private"])
    plt.plot(iters, rmse, "k-x", label="Non-Private", lw=1.5, ms=4, markevery=15)
    for ei, mk in zip(eps5, markers):
        iters, rmse = zip(*res_iter[ei])
        plt.plot(iters, rmse, f"-{mk}", label=f"$\\epsilon_i$={ei}", lw=1.2, ms=4, markevery=15)
    plt.xlabel("Iterations"); plt.ylabel("RMSE")
    plt.title(f"RMSE vs Iterations on\n{label}")
    plt.legend(fontsize=7, ncol=2); plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig1_rmse_iterations_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # =========================================================
    # Exp 2: RMSE & Overall eps vs eps_i  (Fig 4-6)
    # =========================================================
    print(f"\n{'='*50}")
    print(f"  Exp 2: RMSE & Overall eps vs eps_i  (Fig 4-6)")
    print(f"{'='*50}")
    res_eps = {}
    print("  [non-private]")
    m = _fit(20, 0.30, 0.0005, False, tau, V, Rt, Re, Vt, nm, nu, 2500)
    np_rmse = m.rmse_hist[-1][1]
    for ei in eps5:
        print(f"  [eps_i={ei}]")
        m = _fit(20, ei, 0.0005, True, tau, V, Rt, Re, Vt, nm, nu, 2500)
        res_eps[ei] = {"final_rmse": m.rmse_hist[-1][1],
                       "overall_eps": m.overall_epsilon()}

    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()
    ax1.plot(eps5, [res_eps[ei]["final_rmse"] for ei in eps5],
             "s-", color="black", lw=1.5, ms=5)
    ax1.axhline(np_rmse, color="gray", ls=":", lw=1)
    ax2.plot(eps5, [res_eps[ei]["overall_eps"] for ei in eps5],
             "x--", color="black", lw=1.5, ms=6)
    ax1.set_xlabel("$\\epsilon_i$"); ax1.set_ylabel("RMSE")
    ax2.set_ylabel("Overall $\\epsilon$")
    plt.title(f"RMSE and Overall $\\epsilon$ vs $\\epsilon_i$ on\n{label}")
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig4_rmse_overall_eps_ei_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # =========================================================
    # Exp 3: n sweep  (Fig 7-18)
    # NOTE: n capped at 100 for Netflix — n=200/320/500 causes OOM on 8GB GPU
    # =========================================================
    print(f"\n{'='*50}")
    print(f"  Exp 3: n sweep  (Fig 7-18)  [n capped at 100 for Netflix/8GB GPU]")
    print(f"{'='*50}")
    n_vals = [20, 50, 100]   # 200+ causes OOM on Netflix with 8GB VRAM
    res_n  = {"non_private": {}, "private": {ei: {} for ei in eps5},
              "overall_eps": {ei: {} for ei in eps5},
              "time_np": {}, "time_p50": {}}

    for nv in n_vals:
        print(f"  [n={nv} non-private]")
        m = _fit(nv, 0.30, 0.0005, False, tau, V, Rt, Re, Vt, nm, nu, 1000,
                 record_time=True)
        res_n["non_private"][nv] = m.rmse_hist[-1][1]
        res_n["time_np"][nv]     = float(np.mean(m.time_per_iter[:100])) if m.time_per_iter else 0.0
        for ei in eps5:
            print(f"  [n={nv} eps_i={ei}]")
            m = _fit(nv, ei, 0.0005, True, tau, V, Rt, Re, Vt, nm, nu, 1000,
                     record_time=(ei == 0.50))
            res_n["private"][ei][nv]     = m.rmse_hist[-1][1]
            res_n["overall_eps"][ei][nv] = m.overall_epsilon()
            if ei == 0.50:
                res_n["time_p50"][nv] = float(np.mean(m.time_per_iter[:100])) if m.time_per_iter else 0.0

    # Fig 7: RMSE vs n
    plt.figure(figsize=(5.5, 4))
    plt.plot(n_vals, [res_n["non_private"][nv] for nv in n_vals],
             "k-x", label="Non-Private", lw=1.5, ms=4)
    for ei, mk in zip(eps5, markers):
        plt.plot(n_vals, [res_n["private"][ei][nv] for nv in n_vals],
                 f"-{mk}", label=f"$\\epsilon_i$={ei}", lw=1.2, ms=4)
    plt.xlabel("n"); plt.ylabel("RMSE")
    plt.title(f"RMSE vs n on\n{label}")
    plt.legend(fontsize=7); plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig7_rmse_n_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # Fig 10: Overall eps vs n
    plt.figure(figsize=(5.5, 4))
    for ei, mk in zip(eps5, markers):
        plt.plot(n_vals, [res_n["overall_eps"][ei][nv] for nv in n_vals],
                 f"-{mk}", label=f"$\\epsilon_i$={ei}", lw=1.2, ms=4)
    plt.xlabel("n"); plt.ylabel("Overall $\\epsilon$")
    plt.title(f"Overall $\\epsilon$ vs n on\n{label}")
    plt.legend(fontsize=7); plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig10_overall_eps_n_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # Fig 13: RMSE and Overall eps vs n (twin axis)
    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax2 = ax1.twinx()
    ax1.plot(n_vals, [res_n["non_private"][nv] for nv in n_vals],
             "k-^", lw=1.5, ms=4, label="Non-Private")
    for ei, mk in zip(eps5, markers):
        l, = ax1.plot(n_vals, [res_n["private"][ei][nv] for nv in n_vals],
                      f"-{mk}", lw=1.2, ms=4, label=f"$\\epsilon_i$={ei}")
        ax2.plot(n_vals, [res_n["overall_eps"][ei][nv] for nv in n_vals],
                 f"--{mk}", lw=1.0, ms=3, color=l.get_color())
    ax1.set_xlabel("n"); ax1.set_ylabel("RMSE")
    ax2.set_ylabel("Overall $\\epsilon$")
    plt.title(f"RMSE and Overall $\\epsilon$ vs n on\n{label}")
    ax1.legend(fontsize=6); fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig13_rmse_eps_n_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # Fig 16: Time per iteration vs n
    plt.figure(figsize=(5.5, 4))
    plt.plot(n_vals, [res_n["time_np"].get(nv, 0) for nv in n_vals],
             "s--", color="black", lw=1.5, ms=5, label="Non-Private")
    plt.plot(n_vals, [res_n["time_p50"].get(nv, 0) for nv in n_vals],
             "s-",  color="black", lw=1.5, ms=5, label="$\\epsilon_i$=0.50")
    plt.xlabel("n"); plt.ylabel("Time per iteration (ms)")
    plt.title(f"Time per iteration (ms) vs n on\n{label}")
    plt.legend(fontsize=8); plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig16_time_n_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # =========================================================
    # Exp 4: mu sweep  (Fig 19-27)
    # =========================================================
    print(f"\n{'='*50}")
    print(f"  Exp 4: mu sweep  (Fig 19-27)")
    print(f"{'='*50}")
    mu_vals = [0.0005, 0.0008, 0.0010, 0.0013]
    res_mu  = {"non_private": {}, "private": {ei: {} for ei in eps5},
               "overall_eps": {ei: {} for ei in eps5}}

    for mu in mu_vals:
        print(f"  [mu={mu} non-private]")
        m = _fit(20, 0.30, mu, False, tau, V, Rt, Re, Vt, nm, nu, 1000)
        res_mu["non_private"][mu] = m.rmse_hist[-1][1]
        for ei in eps5:
            print(f"  [mu={mu} eps_i={ei}]")
            m = _fit(20, ei, mu, True, tau, V, Rt, Re, Vt, nm, nu, 1000)
            res_mu["private"][ei][mu]     = m.rmse_hist[-1][1]
            res_mu["overall_eps"][ei][mu] = m.overall_epsilon()

    # Fig 19: RMSE vs mu
    plt.figure(figsize=(5.5, 4))
    plt.plot(mu_vals, [res_mu["non_private"][mu] for mu in mu_vals],
             "k-x", label="Non-Private", lw=1.5, ms=4)
    for ei, mk in zip(eps5, markers):
        plt.plot(mu_vals, [res_mu["private"][ei][mu] for mu in mu_vals],
                 f"-{mk}", label=f"$\\epsilon_i$={ei}", lw=1.2, ms=4)
    plt.xlabel("$\\mu$"); plt.ylabel("RMSE")
    plt.title(f"RMSE vs $\\mu$ on\n{label}")
    plt.legend(fontsize=7); plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig19_rmse_mu_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # Fig 22: Overall eps vs mu
    plt.figure(figsize=(5.5, 4))
    for ei, mk in zip(eps5, markers):
        plt.plot(mu_vals, [res_mu["overall_eps"][ei][mu] for mu in mu_vals],
                 f"-{mk}", label=f"$\\epsilon_i$={ei}", lw=1.2, ms=4)
    plt.xlabel("$\\mu$"); plt.ylabel("Overall $\\epsilon$")
    plt.title(f"Overall $\\epsilon$ vs $\\mu$ on\n{label}")
    plt.legend(fontsize=7); plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig22_overall_eps_mu_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    # Fig 25: RMSE and Overall eps vs mu (twin axis)
    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax2 = ax1.twinx()
    ax1.plot(mu_vals, [res_mu["non_private"][mu] for mu in mu_vals],
             "k-^", lw=1.5, ms=4, label="Non-Private")
    for ei, mk in zip(eps5, markers):
        l, = ax1.plot(mu_vals, [res_mu["private"][ei][mu] for mu in mu_vals],
                      f"-{mk}", lw=1.2, ms=4, label=f"$\\epsilon_i$={ei}")
        ax2.plot(mu_vals, [res_mu["overall_eps"][ei][mu] for mu in mu_vals],
                 f"--{mk}", lw=1.0, ms=3, color=l.get_color())
    ax1.set_xlabel("$\\mu$"); ax1.set_ylabel("RMSE")
    ax2.set_ylabel("Overall $\\epsilon$")
    plt.title(f"RMSE and Overall $\\epsilon$ vs $\\mu$ on\n{label}")
    ax1.legend(fontsize=6); fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"fig25_rmse_eps_mu_{name}.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  -> {p}")

    print(f"\n  All Netflix figures saved to: {OUTPUT_DIR}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_pause_controller()

    print("\nLoading Netflix dataset...")
    df = load_netflix(os.path.join(BASE, "archive (1)"))

    print("\nPreprocessing...")
    V_sp, R_sp, nm, nu, user2idx, movie2idx, idx2user, idx2movie, df_clean = \
        preprocess(df)

    tau = float(df_clean["rating"].max() - df_clean["rating"].min())
    print(f"  tau = {tau}")

    print("\nSplitting 90/10 train/test...")
    R_train, R_test, V_test = train_test_split_sparse(V_sp, R_sp)

    print("\nTraining base model (eps_i=0.30, n=20)...")
    base = DPMatrixFactorization(n=20, eps_i=0.30, mu=0.0005,
                                  private=True, max_iter=2500)
    base.fit(V_sp, R_train, R_test, V_test, tau, nm, nu, verbose=True)
    save_model(base, os.path.join(OUTPUT_DIR, "model_netflix_n20_eps0.3.pkl"))
    print(f"  Base RMSE : {base.rmse_hist[-1][1]:.4f}")
    print(f"  Overall e : {base.overall_epsilon():.2f}")

    print("\nRunning all experiments...")
    run_netflix_experiments(V_sp, R_train, R_test, V_test, tau, nm, nu)

    print(f"\nAll done! Figures saved to: {OUTPUT_DIR}")