import numpy as np
import pandas as pd

# === Допоміжні функції ===

def pairwise_euclidean(X):
    """Евклідова відстань"""
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=2))

def pairwise_weighted_euclidean(X, w):
    """Зважена евклідова відстань"""
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(w * (diffs**2), axis=2))

def pairwise_manhattan(X):
    """Мангеттенська (city-block) відстань"""
    diffs = np.abs(X[:, None, :] - X[None, :, :])
    return np.sum(diffs, axis=2)

def jaccard_distance_matrix(B):
    """Відстань Жаккара для бінарних даних"""
    B = (B > 0).astype(int)
    n = B.shape[0]
    M = np.zeros((n, n), dtype=float)
    sums = B.sum(axis=1)
    inter = B @ B.T
    for i in range(n):
        for j in range(n):
            union = sums[i] + sums[j] - inter[i, j]
            M[i, j] = 0.0 if union == 0 else 1.0 - inter[i, j] / union
    return M

def print_matrix(title, M, labels):
    df = pd.DataFrame(np.round(M, 4), index=labels, columns=labels)
    print(f"\n{'='*70}")
    print(title)
    print(df)


# === Завдання 1 ===
x1_t1 = np.array([119.4, 121.0, 16.6, 114.2, 115.8, 15.2, 17.9, 117.5])
x2_t1 = np.array([16.6, 18.1, 15.5, 19.4, 23.2, 16.7, 15.7, 15.2])
X1 = np.vstack([x1_t1, x2_t1]).T
print_matrix("Завдання 1 — Евклідова відстань", pairwise_euclidean(X1), [f"O{i}" for i in range(1, 9)])


# === Завдання 2 ===
x1_t2 = np.array([73.2, 60.2, 63.7, 70.6, 95.1, 75.8, 93.4, 50.5])
x2_t2 = np.array([12.2, 11.6, 1.6, 13.7, 16.1, 11.1, 16.5, 1.2])
X2 = np.vstack([x1_t2, x2_t2]).T
print_matrix("Завдання 2 — Евклідова відстань", pairwise_euclidean(X2), [f"O{i}" for i in range(1, 9)])


# === Завдання 3 ===
x1_t3 = np.array([114.4, 116.0, 11.6, 19.2, 110.8, 11.2, 12.9, 112.5])
x2_t3 = np.array([12.6, 14.1, 12.5, 15.4, 19.2, 11.7, 12.7, 12.2])
X3 = np.vstack([x1_t3, x2_t3]).T
weights = np.array([0.3, 0.7])
print_matrix("Завдання 3 — Зважена Евклідова (w1=0.3, w2=0.7)", pairwise_weighted_euclidean(X3, weights), [f"O{i}" for i in range(1, 9)])


# === Завдання 4 ===
x1_t4 = np.array([133.2, 120.2, 133.7, 120.6, 115.1, 145.8, 153.4, 137.5])
x2_t4 = np.array([24.2, 20.6, 16.6, 36.7, 35.1, 72.1, 56.5, 54.2])
X4 = np.vstack([x1_t4, x2_t4]).T
print_matrix("Завдання 4 — Мангеттенська (City-block) відстань", pairwise_manhattan(X4), [f"O{i}" for i in range(1, 9)])


# === Завдання 5 ===
X5 = np.array([
    [1, 1, 0],
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 1]
])
print_matrix("Завдання 5 — Відстань Жаккара", jaccard_distance_matrix(X5), [f"O{i}" for i in range(1, 6)])
