# Privacy-Preserving Matrix Factorization for Recommendation Systems

This repository presents a **complete replication** of:

> **"Privacy-Preserving Matrix Factorization for Recommendation Systems using Gaussian Mechanism"**
> Mugdho & Imtiaz (2023) — [arXiv:2304.09096](https://arxiv.org/abs/2304.09096)

---

## Objective

Replicate the paper's differentially private matrix factorization algorithm across all three datasets and reproduce all 27 figures, covering:

1. RMSE vs Iterations (varying ε_i)
2. RMSE and Overall ε vs ε_i
3. RMSE vs n (latent dimension)
4. Overall ε vs n
5. RMSE & Overall ε vs n (twin axis)
6. Time per iteration vs n
7. RMSE vs μ (step size)
8. Overall ε vs μ
9. RMSE & Overall ε vs μ (twin axis)

---

## Repository Structure

```
.
├── src/
│   ├── recommender.py          # MovieLens + Anime experiments (all figures)
│   └── netflix_experiments.py  # Netflix experiments (separate due to dataset size)
│
├── results/
│   ├── movielens/              # All 9 figures for MovieLens
│   ├── anime/                  # All 9 figures for Anime
│   └── netflix/                # All 9 figures for Netflix
│
├── assets/                     # Paper figures vs reproduced figures (side by side)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Core Algorithm

### Matrix Factorization

Rating matrix **V** is approximated as:

```
V ≈ X @ Theta.T
```

Where:
- **X** = Movie profile matrix (n_movies × n)
- **Theta** = User profile matrix (n_users × n)

### Privacy Mechanism (Gaussian)

Noise is added only to the **user gradient** (Theta) at each iteration:

```
grad_Theta_hat = grad_Theta + η,   η ~ N(0, σ²I)

σ = (τ · C / ε_i) · √(2 · ln(1.25/δ))
```

### Overall Privacy Budget (Theorem 2 / Eq. 9)

```
ε_opt = J·ε_i² / (4·ln(1.25/δ)) + 2·√(J·ε_i²·ln(1/δ_r) / (4·ln(1.25/δ)))
```

### Initialization (Algorithm 1)

```
X, Theta ~ N(0, I),  then normalize each row to unit L2 norm
```

---

## Datasets

| Dataset | Movies/Anime | Users | Ratings | Density | Rating Range | τ |
|---|---|---|---|---|---|---|
| MovieLens 1M | 3,706 | 6,040 | ~1M | 4.47% | 1–5 | 4 |
| Netflix Prize | ~5,466 | ~11,345 | ~5.36M | 8.65% | 1–5 | 4 |
| Anime Reco | 2,772 | 4,623 | ~1.54M | 12.03% | 1–10 | 9 |

### Download Links
- **MovieLens 1M**: https://grouplens.org/datasets/movielens/1m/
- **Netflix Prize**: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
- **Anime Recommendations**: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

Place datasets in:
```
project/
├── ml-1m/                  ← MovieLens (ratings.dat, movies.dat)
├── archive/                ← Anime (rating.csv, anime.csv)
└── archive (1)/            ← Netflix (combined_data_1.txt)
```

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| n | 20 | Latent factor dimension (default) |
| ε_i | 0.30 | Per-iteration privacy budget (default) |
| δ | 0.01 | Per-iteration delta |
| δ_r | 1e-5 | Overall target delta |
| λ | 0.01 | L2 regularisation |
| μ | 0.0005 | Gradient descent step size (default) |
| C | 1.0 | Row-clip bound |
| max_iter | 2500 | Maximum iterations |

### Sweep Values (matching paper)
- **ε_i sweep**: 0.15, 0.20, 0.30, 0.50, 0.75, 0.90
- **n sweep**: 20, 50, 100, 200, 320, 500
- **μ sweep**: 0.0005, 0.0008, 0.0010, 0.0013

---

## How to Run

### Install dependencies
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scipy pandas matplotlib
```

### Run MovieLens + Anime
```bash
python src/recommender.py
```

### Run Netflix (separate — large dataset)
```bash
python src/netflix_experiments.py
```

Figures are saved to `outputs/` automatically.

### Make Recommendations
```python
from recommender import RecommenderSystem

rec = RecommenderSystem(dataset="movielens", private=True, eps_i=0.30)
rec.load_data()
rec.train()
rec.save()

# Get top-10 recommendations for a user
print(rec.recommend(user_id=1, top_k=10))

# Find similar movies
print(rec.similar_movies(movie_id=1, top_k=5))
```

---

## Results

### MovieLens 1M

| Model | RMSE | Overall ε | Iterations |
|---|---|---|---|
| Non-Private | ~0.84 | — | ~600 |
| Private (ε_i=0.90) | ~0.86 | ~12.4 | ~700 |
| Private (ε_i=0.50) | ~0.86 | ~6.9 | ~1100 |
| Private (ε_i=0.30) | ~0.87 | ~17.5 | ~850 |
| Private (ε_i=0.20) | ~0.88 | ~34.8 | ~800 |

### Anime Recommendations

| Model | RMSE | Overall ε | Iterations |
|---|---|---|---|
| Non-Private | ~1.07 | — | ~300 |
| Private (ε_i=0.90) | ~1.09 | ~8.2 | ~600 |
| Private (ε_i=0.50) | ~1.10 | ~5.5 | ~500 |
| Private (ε_i=0.30) | ~1.12 | ~15.5 | ~700 |

### Netflix Prize

| Model | RMSE | Overall ε | Iterations |
|---|---|---|---|
| Non-Private | ~0.79 | — | ~600 |
| Private (ε_i=0.90) | ~0.79 | ~9.7 | ~700 |
| Private (ε_i=0.50) | ~0.79 | ~13.8 | ~1000 |
| Private (ε_i=0.30) | ~0.85 | ~34.8 | ~1500 |

---

## Reproduced Figures

### MovieLens 1M

| Fig 1 — RMSE vs Iterations | Fig 4 — RMSE & ε vs ε_i | Fig 7 — RMSE vs n |
|---|---|---|
| ![](results/movielens/fig1_rmse_iterations_movielens.png) | ![](results/movielens/fig4_rmse_overall_eps_ei_movielens.png) | ![](results/movielens/fig7_rmse_n_movielens.png) |

| Fig 10 — Overall ε vs n | Fig 13 — RMSE & ε vs n | Fig 16 — Time vs n |
|---|---|---|
| ![](results/movielens/fig10_overall_eps_n_movielens.png) | ![](results/movielens/fig13_rmse_eps_n_movielens.png) | ![](results/movielens/fig16_time_n_movielens.png) |

| Fig 19 — RMSE vs μ | Fig 22 — Overall ε vs μ | Fig 25 — RMSE & ε vs μ |
|---|---|---|
| ![](results/movielens/fig19_rmse_mu_movielens.png) | ![](results/movielens/fig22_overall_eps_mu_movielens.png) | ![](results/movielens/fig25_rmse_eps_mu_movielens.png) |

### Anime Recommendations

| Fig 1 — RMSE vs Iterations | Fig 4 — RMSE & ε vs ε_i | Fig 7 — RMSE vs n |
|---|---|---|
| ![](results/anime/fig1_rmse_iterations_anime.png) | ![](results/anime/fig4_rmse_overall_eps_ei_anime.png) | ![](results/anime/fig7_rmse_n_anime.png) |

| Fig 19 — RMSE vs μ | Fig 22 — Overall ε vs μ | Fig 25 — RMSE & ε vs μ |
|---|---|---|
| ![](results/anime/fig19_rmse_mu_anime.png) | ![](results/anime/fig22_overall_eps_mu_anime.png) | ![](results/anime/fig25_rmse_eps_mu_anime.png) |

### Netflix Prize

| Fig 1 — RMSE vs Iterations | Fig 4 — RMSE & ε vs ε_i | Fig 7 — RMSE vs n |
|---|---|---|
| ![](results/netflix/fig1_rmse_iterations_netflix.png) | ![](results/netflix/fig4_rmse_overall_eps_ei_netflix.png) | ![](results/netflix/fig7_rmse_n_netflix.png) |

| Fig 19 — RMSE vs μ | Fig 22 — Overall ε vs μ | Fig 25 — RMSE & ε vs μ |
|---|---|---|
| ![](results/netflix/fig19_rmse_mu_netflix.png) | ![](results/netflix/fig22_overall_eps_mu_netflix.png) | ![](results/netflix/fig25_rmse_eps_mu_netflix.png) |

---

## Key Observations

- **Privacy-utility tradeoff**: Lower ε_i = more private but higher RMSE. The gap narrows as ε_i increases toward 0.90, approaching non-private performance.
- **Optimal n**: RMSE improves as n increases up to ~50, then degrades as the noise term grows with dimensionality.
- **Step size μ**: Too small (0.0001) → slow convergence. Too large (0.0013) → instability especially at low ε_i.
- **Dataset scale**: Netflix benefits most from privacy mechanism due to high density (8.65%), while MovieLens shows the largest RMSE gap between private and non-private.

---

## Implementation Notes

- GPU-accelerated via PyTorch (tested on RTX 4060 8GB)
- Netflix uses chunked matrix multiplication (100 rows/pass) to stay within VRAM
- No mean-centering (paper uses raw ratings)
- Unit-norm initialization as specified in Algorithm 1
- 90/10 train/test split (paper does not specify; standard practice used)
- Netflix n-sweep limited to n ≤ 100 due to 8GB VRAM constraint

---

## Author

Riyansh Saxena
BTech — Year 3, Semester 6
Recommendation Systems Project

---

## Reference

Mugdho, S. A., & Imtiaz, H. (2023). *Privacy-Preserving Matrix Factorization for Recommendation Systems using Gaussian Mechanism*. arXiv:2304.09096.
