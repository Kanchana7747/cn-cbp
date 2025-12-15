import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1) Load dataset from uploaded file
# ------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")

# Use only numerical features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values


# ------------------------------------------------
# 2) Manual K-means
# ------------------------------------------------
def kmeans(X, k, max_iters=100):
    np.random.seed(42)
    # randomly choose cluster centers
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # assign labels
        labels = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

        # compute new centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # stop if no change
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# ------------------------------------------------
# 3) Manual Silhouette Score
# ------------------------------------------------
def silhouette_score_manual(X, labels):
    n = len(X)
    sil_scores = []

    for i in range(n):
        same = X[labels == labels[i]]
        a = np.mean(np.linalg.norm(same - X[i], axis=1))

        other_labels = [lab for lab in np.unique(labels) if lab != labels[i]]
        b = min(
            np.mean(np.linalg.norm(X[labels == lab] - X[i], axis=1))
            for lab in other_labels
        )

        sil_scores.append((b - a) / max(a, b))

    return np.mean(sil_scores)


# ------------------------------------------------
# 4) Compute silhouette score for k = 2 → 7
# ------------------------------------------------
scores = {}
print("Silhouette scores:")

for k in range(2, 8):
    labels, _ = kmeans(X, k)
    score = silhouette_score_manual(X, labels)
    scores[k] = score
    print(k, score)

# best K
best_k = max(scores, key=scores.get)
print("\n✅ Best K =", best_k)

# ------------------------------------------------
# 5) Final clustering with best k
# ------------------------------------------------
final_labels, final_centroids = kmeans(X, best_k)
df["Cluster"] = final_labels


# ------------------------------------------------
# 6) Plot clusters
# ------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=final_labels)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], marker="X", s=200)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"K-means Result (k={best_k})")
plt.show()

# show dataset with cluster labels
df.head()