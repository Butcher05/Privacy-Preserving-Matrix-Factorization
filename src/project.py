"""
Privacy-Preserving Recommendation System
=========================================
Based on:
  "Privacy-Preserving Matrix Factorization for Recommendation Systems
   using Gaussian Mechanism" - Mugdho & Imtiaz (2023)

CONTROLS (type in PyCharm terminal while training):
  P + Enter  ->  Pause
  R + Enter  ->  Resume
  S + Enter  ->  Save checkpoint and stop

Install dependencies (Anaconda Prompt):
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
CHUNK      = 300   # rows per GPU chunk for large matrices


# =============================================================================
# PAUSE / RESUME CONTROLLER
# =============================================================================

class PauseController:
    """
    Listens for keyboard input in a background thread.
    Type P + Enter to pause, R + Enter to resume, S + Enter to save and stop.
    """
    def __init__(self):
        self.paused    = False
        self.stop_flag = False
        self._thread   = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print("  Controls: P=Pause  R=Resume  S=Save&Stop  (type in terminal + Enter)")

    def _listen(self):
        while True:
            try:
                ch = input().strip().upper()
                if ch == "P" and not self.paused:
                    self.paused = True
                    print("\n  [PAUSED]  Type R to resume, S to save and stop")
                elif ch == "R" and self.paused:
                    self.paused = False
                    print("  [RESUMED]")
                elif ch == "S":
                    self.stop_flag = True
                    self.paused    = False
                    print("  [STOPPING after this iteration and saving...]")
            except EOFError:
                break

    def check(self):
        """
        Call at the end of each iteration.
        Blocks while paused. Returns True if user requested stop.
        """
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
# 1.  DATA LOADERS
# =============================================================================

def load_movielens(folder):
    path = os.path.join(folder, "ratings.dat")
    print(f"Loading MovieLens 1M from: {path}")
    df = pd.read_csv(path, sep="::", engine="python",
                     names=["user", "movie", "rating", "ts"],
                     dtype={"user": int, "movie": int, "rating": float})
    df = df[["user", "movie", "rating"]]
    print(f"  Raw ratings: {len(df):,}")

    titles_path = os.path.join(folder, "movies.dat")
    titles = pd.read_csv(titles_path, sep="::", engine="python",
                         names=["movie", "title", "genre"],
                         encoding="latin-1")[["movie", "title"]]
    return df, titles, "MovieLens 1M"


def load_anime(folder):
    df = pd.read_csv(os.path.join(folder, "rating.csv"))
    df.columns = ["user", "movie", "rating"]
    df = df[df["rating"] != -1]
    print(f"  Raw ratings: {len(df):,}")
    titles = pd.read_csv(os.path.join(folder, "anime.csv"))[["anime_id", "name"]]
    titles.columns = ["movie", "title"]
    return df, titles, "Anime Recommendations"


def load_netflix(folder):
    files = ["combined_data_1.txt", "combined_data_2.txt",
             "combined_data_3.txt", "combined_data_4.txt"]
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
    uc = df["user"].value_counts()
    mc = df["movie"].value_counts()
    df = df[df["user"].isin(uc[(uc >= 500) & (uc <= 2500)].index) &
            df["movie"].isin(mc[mc > 500].index)]
    print(f"  After filter: {len(df):,} ratings | "
          f"{df['user'].nunique():,} users | {df['movie'].nunique():,} movies")
    return df, None, "Netflix Prize"


# =============================================================================
# 2.  PREPROCESSING
# =============================================================================

def preprocess(df, titles=None):
    user_ids  = sorted(df["user"].unique())
    movie_ids = sorted(df["movie"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}

    df = df.copy()
    df["user"]  = df["user"].map(user2idx)
    df["movie"] = df["movie"].map(movie2idx)

    # Paper filters
    uc = df["user"].value_counts()
    mc = df["movie"].value_counts()
    df = df[df["user"].isin(uc[(uc >= 200) & (uc <= 2500)].index) &
            df["movie"].isin(mc[mc > 100].index)]

    # Re-index after filter
    user_ids  = sorted(df["user"].unique())
    movie_ids = sorted(df["movie"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    idx2user  = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}
    df["user"]  = df["user"].map(user2idx)
    df["movie"] = df["movie"].map(movie2idx)

    n_users     = len(user_ids)
    n_movies    = len(movie_ids)
    global_mean = float(df["rating"].mean())
    tau         = float(df["rating"].max() - df["rating"].min())
    mem_gb      = n_movies * n_users * 4 / 1e9
    use_chunked = mem_gb > 4.0

    print(f"  Users   : {n_users:,}")
    print(f"  Movies  : {n_movies:,}")
    print(f"  Ratings : {len(df):,}  (density {len(df)/(n_users*n_movies)*100:.3f}%)")
    print(f"  Mean    : {global_mean:.4f}   tau={tau}")
    print(f"  Dense   : {mem_gb:.2f} GB  ->  chunked={use_chunked}")

    rows = df["movie"].values
    cols = df["user"].values
    vals = (df["rating"].values - global_mean).astype(np.float32)
    ones = np.ones(len(df), dtype=np.float32)

    V_sp = csr_matrix((vals, (rows, cols)), shape=(n_movies, n_users))
    R_sp = csr_matrix((ones, (rows, cols)), shape=(n_movies, n_users))

    # Build title lookup
    title_lookup = {}
    if titles is not None:
        orig2idx = {idx2movie[i]: i for i in range(n_movies)}
        for _, row in titles.iterrows():
            if row["movie"] in orig2idx:
                title_lookup[orig2idx[row["movie"]]] = row["title"]

    return (V_sp, R_sp, tau, global_mean,
            n_movies, n_users, use_chunked,
            user2idx, movie2idx, idx2user, idx2movie,
            title_lookup, df)


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
# 3.  DP MATRIX FACTORIZATION  (Algorithm 1 from paper)
# =============================================================================

class DPMatrixFactorization:
    """
    Differentially Private Matrix Factorization via Gaussian Mechanism.

    Parameters
    ----------
    n        : latent factor dimension        (paper default: 20)
    eps_i    : per-iteration privacy budget   (range: 0.20-0.90)
    delta    : per-iteration delta            (paper: 0.01)
    delta_r  : overall target delta_r         (paper: 1e-5)
    lam      : L2 regularisation             (paper: 0.01)
    mu       : gradient descent step size    (paper: 0.0005)
    C        : row-clip bound               (paper: 1.0)
    private  : False = non-private baseline
    """

    def __init__(self, n=20, eps_i=0.30, delta=0.01, delta_r=1e-5,
                 lam=0.01, mu=0.0005, C=1.0,
                 max_iter=2500, tol=1e-4, private=True, seed=42):
        self.n       = n
        self.eps_i   = eps_i
        self.delta   = delta
        self.delta_r = delta_r
        self.lam     = lam
        self.mu      = mu
        self.C       = C
        self.max_iter = max_iter
        self.tol     = tol
        self.private = private
        self.seed    = seed
        self.X       = None
        self.Theta   = None
        self.rmse_hist = []
        self.J       = 0

    def sigma(self, tau):
        return (tau * self.C / self.eps_i) * np.sqrt(2 * np.log(1.25 / self.delta))

    def overall_epsilon(self):
        J, ei, d, dr = self.J, self.eps_i, self.delta, self.delta_r
        denom = 4 * np.log(1.25 / d)
        return J * ei**2 / denom + 2 * np.sqrt(J * ei**2 * np.log(1 / dr) / denom)

    @staticmethod
    def _clip(M, C):
        norms = torch.linalg.norm(M, dim=1, keepdim=True)
        return M / torch.clamp(norms / C, min=1.0)

    def _rmse(self, V_test_sp, R_test_sp, X, Theta, gmean):
        coo  = R_test_sp.tocoo()
        ri   = torch.tensor(coo.row,            dtype=torch.long,    device=device)
        ci   = torch.tensor(coo.col,            dtype=torch.long,    device=device)
        vv   = torch.tensor(V_test_sp.tocoo().data, dtype=torch.float32, device=device)
        pred = (X[ri] * Theta[ci]).sum(1) + gmean
        true = vv + gmean
        return float(torch.sqrt(((pred - true) ** 2).mean()).item())

    def fit(self, V_sp, R_train_sp, R_test_sp, V_test_sp,
            tau, gmean, n_movies, n_users,
            use_chunked=False, verbose=True):

        torch.manual_seed(self.seed)
        sig   = float(self.sigma(tau)) if self.private else 0.0
        X     = torch.randn(n_movies, self.n, device=device) * 0.01
        Theta = torch.randn(n_users,  self.n, device=device) * 0.01

        prev_rmse = float("inf")
        t0        = time.time()

        # Load dense matrices once (if they fit in VRAM)
        if not use_chunked:
            V_d = torch.tensor(V_sp.toarray(),       dtype=torch.float32, device=device)
            R_d = torch.tensor(R_train_sp.toarray(), dtype=torch.float32, device=device)

        for t in range(1, self.max_iter + 1):

            # ---- gradient computation ----
            if use_chunked:
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
            else:
                res = (X @ Theta.T - V_d) * R_d
                gX  = res @ self._clip(Theta, self.C) + self.lam * X
                gTh = res.T @ self._clip(X, self.C)   + self.lam * Theta

            # ---- add noise (privacy mechanism) ----
            if self.private:
                gTh += torch.randn_like(gTh) * sig

            # ---- gradient descent update ----
            X     = X     - self.mu * gX
            Theta = Theta - self.mu * gTh

            # ---- evaluate every 50 iters ----
            if t % 50 == 0 or t == 1:
                with torch.no_grad():
                    rmse = self._rmse(V_test_sp, R_test_sp, X, Theta, gmean)
                self.rmse_hist.append((t, rmse))

                tag = f"eps_i={self.eps_i}" if self.private else "non-private"
                if verbose and t % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"    iter {t:5d} [{tag}]  RMSE={rmse:.4f}  ({elapsed:.0f}s)")

                # Converged?
                if abs(prev_rmse - rmse) < self.tol and t > 200 and rmse < 2.0:
                    if verbose:
                        elapsed = time.time() - t0
                        print(f"    Converged iter {t} [{tag}]  RMSE={rmse:.4f}  ({elapsed:.0f}s)")
                    self.J = t
                    break
                prev_rmse = rmse

                # ---- pause / save-and-stop check ----
                ctrl = get_pause_ctrl()
                if ctrl is not None:
                    should_stop = ctrl.check()
                    if should_stop:
                        print(f"    Stopped at iter {t}  RMSE={rmse:.4f}")
                        self.J     = t
                        self.X     = X.cpu().numpy()
                        self.Theta = Theta.cpu().numpy()
                        if not use_chunked:
                            del V_d, R_d
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        return self
        else:
            self.J = self.max_iter

        self.X     = X.cpu().numpy()
        self.Theta = Theta.cpu().numpy()
        if not use_chunked:
            del V_d, R_d
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self


# =============================================================================
# 4.  RECOMMENDER SYSTEM
# =============================================================================

class RecommenderSystem:
    """
    Full recommendation pipeline.

    Usage
    -----
    rec = RecommenderSystem("movielens")
    rec.train()                          # or rec.load("model.pkl")
    rec.recommend(user_id=1, top_k=10)
    rec.similar_movies(movie_id=1)
    rec.evaluate()
    rec.save("model.pkl")
    """

    def __init__(self, dataset="movielens", private=True, eps_i=0.30,
                 n=20, mu=0.0005, max_iter=2500):
        self.dataset    = dataset
        self.private    = private
        self.eps_i      = eps_i
        self.n          = n
        self.mu         = mu
        self.max_iter   = max_iter
        self.model      = None
        self.is_trained = False
        # set by load_data()
        self.V_sp = self.R_sp = None
        self.tau  = self.gmean = None
        self.n_movies = self.n_users = None
        self.use_chunked = False
        self.R_train = self.R_test = self.V_test = None
        self.user2idx = self.movie2idx = None
        self.idx2user = self.idx2movie = None
        self.title_lookup = {}
        self.label = ""

    # ------------------------------------------------------------------
    def load_data(self):
        print(f"\nLoading {self.dataset}...")
        if self.dataset == "movielens":
            df, titles, label = load_movielens(os.path.join(BASE, "ml-1m"))
        elif self.dataset == "anime":
            df, titles, label = load_anime(os.path.join(BASE, "archive"))
        elif self.dataset == "netflix":
            df, titles, label = load_netflix(os.path.join(BASE, "archive (1)"))
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        self.label = label

        print("Preprocessing...")
        (self.V_sp, self.R_sp, self.tau, self.gmean,
         self.n_movies, self.n_users, self.use_chunked,
         self.user2idx, self.movie2idx,
         self.idx2user, self.idx2movie,
         self.title_lookup, self.df) = preprocess(df, titles)

        print("Splitting 90/10 train/test...")
        self.R_train, self.R_test, self.V_test = \
            train_test_split_sparse(self.V_sp, self.R_sp)

    # ------------------------------------------------------------------
    def train(self):
        if self.V_sp is None:
            self.load_data()
        tag = "Private" if self.private else "Non-Private"
        print(f"\nTraining {tag} MF  (n={self.n}, eps_i={self.eps_i}, mu={self.mu})...")

        self.model = DPMatrixFactorization(
            n=self.n, eps_i=self.eps_i, mu=self.mu,
            private=self.private, max_iter=self.max_iter
        )
        self.model.fit(
            self.V_sp, self.R_train, self.R_test, self.V_test,
            self.tau, self.gmean, self.n_movies, self.n_users,
            use_chunked=self.use_chunked
        )
        self.is_trained = True

        rmse = self.model.rmse_hist[-1][1]
        print(f"\n{'='*45}")
        print(f"  Training complete!")
        print(f"  Final RMSE      : {rmse:.4f}")
        print(f"  Iterations (J)  : {self.model.J}")
        if self.private:
            print(f"  Overall epsilon : {self.model.overall_epsilon():.2f}")
        print(f"{'='*45}")

    # ------------------------------------------------------------------
    def save(self, path=None):
        """Save model to disk. Next run loads instantly — no retraining."""
        assert self.is_trained, "Call train() first"
        if path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR,
                                f"model_{self.dataset}_n{self.n}_eps{self.eps_i}.pkl")
        data = {
            "X":           self.model.X,
            "Theta":       self.model.Theta,
            "rmse_hist":   self.model.rmse_hist,
            "J":           self.model.J,
            "gmean":       self.gmean,
            "tau":         self.tau,
            "n_movies":    self.n_movies,
            "n_users":     self.n_users,
            "user2idx":    self.user2idx,
            "movie2idx":   self.movie2idx,
            "idx2user":    self.idx2user,
            "idx2movie":   self.idx2movie,
            "title_lookup": self.title_lookup,
            "label":       self.label,
            "dataset":     self.dataset,
            "private":     self.private,
            "eps_i":       self.eps_i,
            "n":           self.n,
            "mu":          self.mu,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Model saved -> {path}  ({size_mb:.1f} MB)")
        return path

    # ------------------------------------------------------------------
    def load(self, path):
        """Load a saved model — skips training entirely."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model           = DPMatrixFactorization(
            n=data["n"], eps_i=data["eps_i"],
            mu=data["mu"], private=data["private"]
        )
        self.model.X         = data["X"]
        self.model.Theta     = data["Theta"]
        self.model.rmse_hist = data["rmse_hist"]
        self.model.J         = data["J"]
        self.gmean           = data["gmean"]
        self.tau             = data["tau"]
        self.n_movies        = data["n_movies"]
        self.n_users         = data["n_users"]
        self.user2idx        = data["user2idx"]
        self.movie2idx       = data["movie2idx"]
        self.idx2user        = data["idx2user"]
        self.idx2movie       = data["idx2movie"]
        self.title_lookup    = data["title_lookup"]
        self.label           = data["label"]
        self.dataset         = data["dataset"]
        self.private         = data["private"]
        self.eps_i           = data["eps_i"]
        self.n               = data["n"]
        self.mu              = data["mu"]
        self.is_trained      = True
        rmse = self.model.rmse_hist[-1][1]
        print(f"  Model loaded <- {path}")
        print(f"  RMSE={rmse:.4f}   Iterations={self.model.J}")
        return self

    # ------------------------------------------------------------------
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a (user_id, movie_id) pair."""
        assert self.is_trained, "Call train() first"
        if user_id not in self.user2idx or movie_id not in self.movie2idx:
            return None
        ui    = self.user2idx[user_id]
        mi    = self.movie2idx[movie_id]
        score = float(np.dot(self.model.X[mi], self.model.Theta[ui])) + self.gmean
        return float(np.clip(score, 1.0, 5.0))

    # ------------------------------------------------------------------
    def recommend(self, user_id, top_k=10, exclude_seen=True):
        """Top-K movie recommendations for a user."""
        assert self.is_trained, "Call train() first"
        if user_id not in self.user2idx:
            print(f"User {user_id} not found")
            return None

        ui     = self.user2idx[user_id]
        scores = self.model.X @ self.model.Theta[ui] + self.gmean
        scores = np.clip(scores, 1.0, 5.0)

        if exclude_seen and self.R_sp is not None:
            R_coo       = self.R_sp.tocoo()
            seen_movies = set(R_coo.row[R_coo.col == ui])
            for mi in seen_movies:
                scores[mi] = -np.inf

        top_idx = np.argsort(scores)[::-1][:top_k]
        rows = []
        for rank, mi in enumerate(top_idx, 1):
            orig_id = self.idx2movie[mi]
            title   = self.title_lookup.get(mi, f"Movie_{orig_id}")
            rows.append({
                "rank":             rank,
                "movie_id":         orig_id,
                "title":            title,
                "predicted_rating": round(float(scores[mi]), 2)
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def similar_movies(self, movie_id, top_k=10):
        """Movies most similar to a given movie (cosine similarity)."""
        assert self.is_trained, "Call train() first"
        if movie_id not in self.movie2idx:
            print(f"Movie {movie_id} not found")
            return None

        mi        = self.movie2idx[movie_id]
        mvec      = self.model.X[mi]
        norms     = np.linalg.norm(self.model.X, axis=1) + 1e-12
        mvec_norm = np.linalg.norm(mvec) + 1e-12
        sims      = (self.model.X @ mvec) / (norms * mvec_norm)
        sims[mi]  = -np.inf

        top_idx = np.argsort(sims)[::-1][:top_k]
        rows = []
        for rank, idx in enumerate(top_idx, 1):
            orig_id = self.idx2movie[idx]
            title   = self.title_lookup.get(idx, f"Movie_{orig_id}")
            rows.append({"rank": rank, "movie_id": orig_id,
                         "title": title, "similarity": round(float(sims[idx]), 4)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def evaluate(self):
        """Print test RMSE."""
        assert self.is_trained, "Call train() first"
        rmse = self.model.rmse_hist[-1][1]
        print(f"\n  Test RMSE : {rmse:.4f}")
        if self.private:
            print(f"  Overall e : {self.model.overall_epsilon():.2f}")
        return rmse

    # ------------------------------------------------------------------
    def plot_training_curve(self, save_path=None):
        """Plot RMSE vs iterations."""
        assert self.is_trained, "Call train() first"
        iters, rmse = zip(*self.model.rmse_hist)
        plt.figure(figsize=(8, 4))
        tag = f"eps_i={self.eps_i}" if self.private else "Non-Private"
        plt.plot(iters, rmse, label=tag, lw=1.5)
        plt.xlabel("Iterations"); plt.ylabel("RMSE")
        plt.title(f"Training Curve - {self.label}")
        plt.legend(); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path}")
        plt.close()


# =============================================================================
# 5.  PAPER EXPERIMENTS
# =============================================================================

def run_experiments(rec):
    """Reproduce all 3 paper experiments and save plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name  = rec.dataset
    label = rec.label
    V, Rt, Re, Vt = rec.V_sp, rec.R_train, rec.R_test, rec.V_test
    tau, gm, nm, nu, uc = rec.tau, rec.gmean, rec.n_movies, rec.n_users, rec.use_chunked

    def fit(n, ei, mu, private, max_iter):
        m = DPMatrixFactorization(n=n, eps_i=ei, mu=mu,
                                   private=private, max_iter=max_iter)
        m.fit(V, Rt, Re, Vt, tau, gm, nm, nu, use_chunked=uc)
        return m

    eps_vals = [0.20, 0.30, 0.50, 0.75, 0.90]

    # ---- Experiment 1: eps_i sweep ----
    print(f"\n--- Experiment 1: eps_i sweep ---")
    res_eps = {}
    print("  [non-private]")
    m = fit(20, 0.30, 0.0005, False, 1500)
    res_eps["non_private"] = m.rmse_hist
    for ei in eps_vals:
        print(f"  [eps_i={ei}]")
        m = fit(20, ei, 0.0005, True, 1500)
        res_eps[ei] = {"rmse_hist": m.rmse_hist,
                       "final_rmse": m.rmse_hist[-1][1],
                       "overall_epsilon": m.overall_epsilon()}

    plt.figure(figsize=(9, 5))
    iters, rmse = zip(*res_eps["non_private"])
    plt.plot(iters, rmse, "k--", label="Non-Private", lw=2)
    for ei in eps_vals:
        iters, rmse = zip(*res_eps[ei]["rmse_hist"])
        plt.plot(iters, rmse, label=f"eps={ei}", lw=1.3)
    plt.xlabel("Iterations"); plt.ylabel("RMSE")
    plt.title(f"RMSE vs Iterations - {label}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig1_rmse_iter_{name}.png"), dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()
    ax1.plot(eps_vals, [res_eps[ei]["final_rmse"]      for ei in eps_vals], "bs-",  lw=1.5)
    ax2.plot(eps_vals, [res_eps[ei]["overall_epsilon"] for ei in eps_vals], "rx--", lw=1.5)
    ax1.set_xlabel("eps_i"); ax1.set_ylabel("RMSE", color="b")
    ax2.set_ylabel("Overall eps", color="r")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2.tick_params(axis="y", labelcolor="r")
    plt.title(f"RMSE & Overall eps vs eps_i - {label}"); fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig2_rmse_eps_{name}.png"), dpi=150)
    plt.close()
    print(f"  Saved fig1 and fig2")

    # ---- Experiment 2: n sweep ----
    print(f"\n--- Experiment 2: n sweep ---")
    n_vals = [5, 10, 20, 50, 100, 200]  # 320 excluded: unstable at mu=0.0005
    res_n  = {"non_private": {}, "private": {ei: {} for ei in eps_vals}}
    for nv in n_vals:
        print(f"  [n={nv}]")
        iters = 600 if nv >= 100 else 1000  # large n needs smaller step
        m = fit(nv, 0.30, 0.0005, False, iters)
        res_n["non_private"][nv] = m.rmse_hist[-1][1]
        for ei in eps_vals:
            m = fit(nv, ei, 0.0005, True, iters)
            res_n["private"][ei][nv] = m.rmse_hist[-1][1]

    plt.figure(figsize=(9, 5))
    plt.plot(n_vals, [res_n["non_private"][nv] for nv in n_vals], "k--", label="Non-Private", lw=2)
    for ei in eps_vals:
        plt.plot(n_vals, [res_n["private"][ei][nv] for nv in n_vals], label=f"eps={ei}", lw=1.3)
    plt.xlabel("n (latent dimension)"); plt.ylabel("RMSE")
    plt.title(f"RMSE vs n - {label}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig3_rmse_n_{name}.png"), dpi=150)
    plt.close()
    print(f"  Saved fig3")

    # ---- Experiment 3: mu sweep ----
    print(f"\n--- Experiment 3: mu sweep ---")
    mu_vals = [0.0001, 0.0003, 0.0005, 0.0008]
    res_mu  = {"non_private": {}, "private": {ei: {} for ei in eps_vals}}
    for mu in mu_vals:
        print(f"  [mu={mu}]")
        m = fit(20, 0.30, mu, False, 1000)
        res_mu["non_private"][mu] = m.rmse_hist[-1][1]
        for ei in eps_vals:
            m = fit(20, ei, mu, True, 1000)
            res_mu["private"][ei][mu] = m.rmse_hist[-1][1]

    plt.figure(figsize=(9, 5))
    plt.plot(mu_vals, [res_mu["non_private"][mu] for mu in mu_vals], "k--", label="Non-Private", lw=2)
    for ei in eps_vals:
        plt.plot(mu_vals, [res_mu["private"][ei][mu] for mu in mu_vals], label=f"eps={ei}", lw=1.3)
    plt.xlabel("Step size mu"); plt.ylabel("RMSE")
    plt.title(f"RMSE vs mu - {label}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig4_rmse_mu_{name}.png"), dpi=150)
    plt.close()
    print(f"  Saved fig4")

    print(f"\n  All plots saved to: {OUTPUT_DIR}")


# =============================================================================
# 6.  MAIN
# =============================================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Start pause/resume controller
    start_pause_controller()

    # -----------------------------------------------------------------------
    # STEP 1: Train (or load saved) model on MovieLens
    # -----------------------------------------------------------------------
    MODEL_PATH = os.path.join(OUTPUT_DIR, "model_movielens_n20_eps0.3.pkl")

    rec = RecommenderSystem(
        dataset  = "movielens",
        private  = True,
        eps_i    = 0.30,
        n        = 20,
        mu       = 0.0005,
        max_iter = 2500
    )

    if os.path.exists(MODEL_PATH):
        print("\nFound saved model - loading (no retraining needed)")
        rec.load(MODEL_PATH)
    else:
        rec.load_data()
        rec.train()
        rec.save(MODEL_PATH)

    rec.evaluate()

    # -----------------------------------------------------------------------
    # STEP 2: Sample recommendations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 50)

    sample_users = list(rec.user2idx.keys())[:3]
    for uid in sample_users:
        print(f"\nTop 10 for User {uid}:")
        recs = rec.recommend(user_id=uid, top_k=10)
        if recs is not None:
            print(recs.to_string(index=False))

    # -----------------------------------------------------------------------
    # STEP 3: Similar movies
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("SIMILAR MOVIES")
    print("=" * 50)

    sample_movies = list(rec.movie2idx.keys())[:3]
    for mid in sample_movies:
        title = rec.title_lookup.get(rec.movie2idx[mid], f"Movie_{mid}")
        print(f"\nSimilar to '{title}':")
        sims = rec.similar_movies(movie_id=mid, top_k=5)
        if sims is not None:
            print(sims.to_string(index=False))

    # -----------------------------------------------------------------------
    # STEP 4: Training curve plot
    # -----------------------------------------------------------------------
    rec.plot_training_curve(
        save_path=os.path.join(OUTPUT_DIR, "training_curve_movielens.png"))

    # -----------------------------------------------------------------------
    # STEP 5: Paper experiments on all 3 datasets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PAPER EXPERIMENTS - ALL 3 DATASETS")
    print("=" * 50)

    for ds in ["movielens", "anime", "netflix"]:
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'='*60}")
        r = RecommenderSystem(dataset=ds, private=True,
                              eps_i=0.30, n=20, mu=0.0005, max_iter=2500)
        r.load_data()
        r.train()
        r.save()
        r.evaluate()
        run_experiments(r)

    print(f"\nAll done! Results saved to: {OUTPUT_DIR}")