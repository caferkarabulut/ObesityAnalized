import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def run_classification(csv_path: str) -> None:
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()

    categorical_cols = [
        "Gender", "family_history_with_overweight", "FAVC", "CAEC",
        "SMOKE", "SCC", "CALC", "MTRANS"
    ]

    # Encode categorical columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["NObeyesdad"])
    class_names = target_encoder.classes_

    X = df.drop("NObeyesdad", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    precision_scores = [report[c]["precision"] for c in class_names]
    recall_scores = [report[c]["recall"] for c in class_names]
    f1_scores = [report[c]["f1-score"] for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion Matrix
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Tahmin Edilen", fontsize=12)
    axes[0].set_ylabel("Gerçek Değer", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # Class metrics bar chart
    x = np.arange(len(class_names))
    width = 0.25

    axes[1].bar(x - width, precision_scores, width, label="Precision", color="#3498db")
    axes[1].bar(x, recall_scores, width, label="Recall", color="#2ecc71")
    axes[1].bar(x + width, f1_scores, width, label="F1-Score", color="#e74c3c")

    axes[1].set_xlabel("Sınıflar", fontsize=12)
    axes[1].set_ylabel("Skor", fontsize=12)
    axes[1].set_title("Sınıf Bazlı Performans Metrikleri", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("confusion_matrix_ve_metrikler.png", dpi=150, bbox_inches="tight")
    plt.show()
