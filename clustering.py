import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
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

    # =====================================================================
    # HIERARCHICAL CLUSTERING (Hiyerarşik Kümeleme)
    # =====================================================================

    # Agglomerative Clustering
    hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    df["Hierarchical_Label"] = hierarchical.fit_predict(X_scaled)
    
    hierarchical_silhouette = silhouette_score(X_scaled, df["Hierarchical_Label"])
    print(f"\nHierarchical Clustering Silhouette Skoru (k={best_k}): {hierarchical_silhouette:.4f}")

    # =====================================================================
    # DBSCAN (Yoğunluk Tabanlı Kümeleme)
    # =====================================================================
    
    # DBSCAN - eps ve min_samples parametreleri
    dbscan = DBSCAN(eps=2.5, min_samples=5)
    df["DBSCAN_Label"] = dbscan.fit_predict(X_scaled)
    
    n_clusters_dbscan = len(set(df["DBSCAN_Label"])) - (1 if -1 in df["DBSCAN_Label"].values else 0)
    n_noise = (df["DBSCAN_Label"] == -1).sum()
    
    print(f"\nDBSCAN Sonuçları:")
    print(f"  - Bulunan küme sayısı: {n_clusters_dbscan}")
    print(f"  - Gürültü (noise) nokta sayısı: {n_noise}")
    
    # DBSCAN için silhouette (gürültü noktaları hariç)
    if n_clusters_dbscan > 1:
        mask_no_noise = df["DBSCAN_Label"] != -1
        if mask_no_noise.sum() > 0:
            dbscan_silhouette = silhouette_score(X_scaled[mask_no_noise], df.loc[mask_no_noise, "DBSCAN_Label"])
            print(f"  - Silhouette Skoru (gürültü hariç): {dbscan_silhouette:.4f}")
    
    # =====================================================================
    # ÜÇ YÖNTEMİN KARŞILAŞTIRMALI GÖRSELLEŞTİRMESİ
    # =====================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # K-Means
    for i in range(best_k):
        mask = df["Cluster_Label"] == i
        axes[0].scatter(
            df.loc[mask, "PC1"],
            df.loc[mask, "PC2"],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f"Küme {i}",
            alpha=0.6,
            s=50,
            edgecolors="white"
        )
    axes[0].set_xlabel("PC1", fontsize=11)
    axes[0].set_ylabel("PC2", fontsize=11)
    axes[0].set_title("K-Means Kümeleme", fontsize=13, fontweight="bold")
    axes[0].legend(title="Kümeler", loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Hierarchical Clustering
    for i in range(best_k):
        mask = df["Hierarchical_Label"] == i
        axes[1].scatter(
            df.loc[mask, "PC1"],
            df.loc[mask, "PC2"],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f"Küme {i}",
            alpha=0.6,
            s=50,
            edgecolors="white"
        )
    axes[1].set_xlabel("PC1", fontsize=11)
    axes[1].set_ylabel("PC2", fontsize=11)
    axes[1].set_title("Hiyerarşik Kümeleme (Ward)", fontsize=13, fontweight="bold")
    axes[1].legend(title="Kümeler", loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # DBSCAN
    unique_labels = sorted(df["DBSCAN_Label"].unique())
    for i, label in enumerate(unique_labels):
        mask = df["DBSCAN_Label"] == label
        if label == -1:
            # Gürültü noktaları
            axes[2].scatter(
                df.loc[mask, "PC1"],
                df.loc[mask, "PC2"],
                c="gray",
                marker="x",
                label="Gürültü",
                alpha=0.4,
                s=30
            )
        else:
            axes[2].scatter(
                df.loc[mask, "PC1"],
                df.loc[mask, "PC2"],
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=f"Küme {label}",
                alpha=0.6,
                s=50,
                edgecolors="white"
            )
    axes[2].set_xlabel("PC1", fontsize=11)
    axes[2].set_ylabel("PC2", fontsize=11)
    axes[2].set_title(f"DBSCAN (eps=2.5, min_samples=5)", fontsize=13, fontweight="bold")
    axes[2].legend(title="Kümeler", loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Kümeleme Yöntemlerinin Karşılaştırması", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("clustering_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Özet istatistikler
    print("\n" + "="*60)
    print("KÜMELEME YÖNTEMLERİ KARŞILAŞTIRMASI")
    print("="*60)
    print(f"\n1. K-Means:")
    print(f"   - Küme sayısı: {best_k}")
    print(f"   - Silhouette Skoru: {silhouettes[best_k-2]:.4f}")
    
    print(f"\n2. Hiyerarşik Kümeleme (Ward):")
    print(f"   - Küme sayısı: {best_k}")
    print(f"   - Silhouette Skoru: {hierarchical_silhouette:.4f}")
    
    print(f"\n3. DBSCAN:")
    print(f"   - Bulunan küme sayısı: {n_clusters_dbscan}")
    print(f"   - Gürültü nokta sayısı: {n_noise}")
    if n_clusters_dbscan > 1 and mask_no_noise.sum() > 0:
        print(f"   - Silhouette Skoru: {dbscan_silhouette:.4f}")
    print("="*60)
