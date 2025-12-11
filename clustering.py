import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def run_clustering(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # BMI
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    categorical_cols = [
        "Gender", "family_history_with_overweight", "FAVC", "CAEC",
        "SMOKE", "SCC", "CALC", "MTRANS"
    ]

    # Encode categorical
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    cluster_features = [
        "Age", "Height", "Weight", "BMI", "Gender",
        "family_history_with_overweight", "FAVC", "FCVC", "NCP",
        "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
    ]

    X_cluster = df[cluster_features]
    X_scaled = StandardScaler().fit_transform(X_cluster)

    # Elbow + Silhouette
    k_range = range(2, 9)
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    best_k = list(k_range)[int(np.argmax(silhouettes))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Küme Sayısı (k)", fontsize=12)
    axes[0].set_ylabel("Inertia", fontsize=12)
    axes[0].set_title("Dirsek (Elbow) Yöntemi", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=best_k, color="r", linestyle="--", label=f"Seçilen k={best_k}")
    axes[0].legend()

    axes[1].plot(list(k_range), silhouettes, "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Küme Sayısı (k)", fontsize=12)
    axes[1].set_ylabel("Silhouette Skoru", fontsize=12)
    axes[1].set_title("Silhouette Skoru Analizi", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=best_k, color="r", linestyle="--", label=f"Optimal k={best_k}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("kmeans_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Final KMeans
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster_Label"] = kmeans_final.fit_predict(X_scaled)

    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e91e63", "#00bcd4"]
    markers = ["o", "s", "^", "D", "v", "p", "h", "*"]

    for i in range(best_k):
        mask = df["Cluster_Label"] == i
        ax.scatter(
            df.loc[mask, "PC1"],
            df.loc[mask, "PC2"],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f"Küme {i}",
            alpha=0.6,
            s=60,
            edgecolors="white"
        )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("K-Means Kümelerinin PCA Üzerinde Görselleştirilmesi", fontsize=14, fontweight="bold")
    ax.legend(title="Kümeler", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kmeans_pca_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
