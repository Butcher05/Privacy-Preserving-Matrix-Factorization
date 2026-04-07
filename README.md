# Privacy-Preserving Matrix Factorization for Recommendation Systems

Replication of:
> **"Privacy-Preserving Matrix Factorization for Recommendation Systems using Gaussian Mechanism"**
> Mugdho & Imtiaz (2023) — [arXiv:2304.09096](https://arxiv.org/abs/2304.09096)

---

## Repository Structure

```
.
├── src/
│   ├── recommender.py            # MovieLens + Anime — training, experiments, recommendations
│   └── netflix_experiments.py   # Netflix — separate script (large dataset)
│
├── results/                      # All figures (paper screenshots + our reproductions)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Core Algorithm

**V ≈ X @ Theta.T** — Rating matrix decomposed into movie profiles X and user profiles Theta.

Gaussian noise added only to the **user gradient** each iteration to protect user privacy:

```
σ = (τ · C / ε_i) · √(2 · ln(1.25/δ))
grad_Theta_hat = grad_Theta + η,   η ~ N(0, σ²I)
```

Overall privacy budget (Theorem 2 / Eq. 9):
```
ε_opt = J·ε_i² / [4·ln(1.25/δ)] + 2·√(J·ε_i²·ln(1/δ_r) / [4·ln(1.25/δ)])
```

---

## Datasets

| Dataset | Movies | Users | Ratings | Density | Rating Range | τ |
|---|---|---|---|---|---|---|
| MovieLens 1M | 3,706 | 6,040 | ~1M | 4.47% | 1–5 | 4 |
| Netflix Prize | ~5,466 | ~11,345 | ~5.36M | 8.65% | 1–5 | 4 |
| Anime Reco | 2,772 | 4,623 | ~1.54M | 12.03% | 1–10 | 9 |

---

## Parameters

| Parameter | Default | Sweep values |
|---|---|---|
| n | 20 | 20, 50, 100, 200, 320, 500 |
| ε_i | 0.30 | 0.15, 0.20, 0.30, 0.50, 0.75, 0.90 |
| μ | 0.0005 | 0.0005, 0.0008, 0.0010, 0.0013 |
| δ | 0.01 | — |
| δ_r | 1e-5 | — |
| λ | 0.01 | — |
| C | 1.0 | — |

---

## How to Run

```bash
# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scipy pandas matplotlib

# Run MovieLens + Anime
python src/recommender.py

# Run Netflix (separate — large dataset)
python src/netflix_experiments.py
```

### Make Recommendations
```python
from recommender import RecommenderSystem

rec = RecommenderSystem(dataset="movielens", private=True, eps_i=0.30)
rec.load_data()
rec.train()
rec.save()

rec.recommend(user_id=1, top_k=10)       # Top-10 movies for a user
rec.similar_movies(movie_id=1, top_k=5)  # Movies similar to a given movie
```

---

## Results — Paper vs Ours

> **Left = Paper (Mugdho & Imtiaz, 2023) &nbsp;|&nbsp; Right = Our Reproduction**

---

## Fig 1–3 | RMSE vs Iterations

*RMSE decreases over iterations. Lower ε_i = more noise = slower convergence and higher final RMSE.*

### Fig 1 — MovieLens 1M
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103533.png) | ![](results/fig1_rmse_iter_movielens.png) |

### Fig 2 — Netflix dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103538.png) | ![](results/fig1_rmse_iterations_netflix.png) |

### Fig 3 — AnimeReco dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103542.png) | ![](results/fig1_rmse_iter_anime.png) |

---

## Fig 4–6 | RMSE and Overall ε vs ε_i

*As ε_i increases: RMSE falls (better accuracy) while overall ε rises (less private). Classic privacy-utility tradeoff.*

### Fig 4 — MovieLens 1M
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20100814.png) | ![](results/fig2_rmse_eps_movielens.png) |

### Fig 5 — Netflix dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20100818.png) | ![](results/fig2_rmse_eps_netflix.png) |

### Fig 6 — AnimeReco dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20100824.png) | ![](results/fig2_rmse_eps_anime.png) |

---

## Fig 7–9 | RMSE vs n

*RMSE first improves as n increases (richer model), then degrades as noise grows with matrix size.*

### Fig 7 — MovieLens 1M
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103642.png) | ![](results/fig3_rmse_n_movielens.png) |

### Fig 8 — Netflix dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103646.png) | ![](results/fig7_rmse_n_netflix.png) |

### Fig 9 — AnimeReco dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103649.png) | ![](results/fig3_rmse_n_anime.png) |

---

## Fig 10–12 | Overall ε vs n

*Overall privacy budget grows with n — larger latent dimension needs more iterations, accumulating more privacy cost.*

### Fig 11 — Netflix dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103722.png) | ![](results/fig10_overall_eps_n_netflix.png) |

---

## Fig 13–15 | RMSE and Overall ε vs n

*Combined view: RMSE (solid) and overall ε (dashed) vs n on same plot.*

### Fig 14 — Netflix dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103822.png) | ![](results/fig13_rmse_eps_n_netflix.png) |

---

## Fig 16–18 | Time per Iteration vs n

*Computation time scales with n. Private and non-private are nearly identical — noise addition is negligible vs matrix multiplication.*

### Fig 17 — Netflix dataset
| Paper | Ours |
|:---:|:---:|
| ![](results/Screenshot%202026-04-07%20103835.png) | ![](results/fig16_time_n_netflix.png) |

---

## Key Observations

**Fig 1–3 (RMSE vs Iterations):** Our reproduction captures the same convergence shape. RMSE values are slightly higher than the paper due to dataset size differences and the 90/10 test split penalising generalisation.

**Fig 4–6 (Privacy-Utility Tradeoff):** Both paper and ours show the same X-shaped crossing of RMSE and overall ε. The tradeoff is faithfully reproduced across all three datasets.

**Fig 7–9 (RMSE vs n):** Paper shows RMSE decreasing monotonically (sign of train-set overfitting). Our results show RMSE rising after n=20–50, reflecting proper test-set generalisation behaviour.

**Fig 10–12 (Overall ε vs n):** Paper shows ε rising with n. Our Netflix result is flat because n was capped at 100 due to 8GB VRAM constraint.

**Fig 16–18 (Time vs n):** Paper shows linear growth. Our Netflix times are flatter due to chunked implementation.

---

## Implementation Notes

- GPU-accelerated via PyTorch (NVIDIA RTX 4060 8GB)
- Netflix uses chunked matrix multiplication (100 rows/pass) to stay within 8GB VRAM
- Raw ratings — no mean-centering (matches paper)
- Unit-norm row initialization as per Algorithm 1
- 90/10 train/test split (paper does not specify; standard practice applied)
- Netflix n-sweep limited to n ≤ 100 due to GPU memory constraint

---

## Author

**Riyansh Saxena**
BTech — Year 3, Semester 6
Recommendation Systems Project

---

## Reference

Mugdho, S. A., & Imtiaz, H. (2023). *Privacy-Preserving Matrix Factorization for Recommendation Systems using Gaussian Mechanism*. arXiv:2304.09096.
