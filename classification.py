import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

    # SVM için ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # =====================================================================
    # RANDOM FOREST
    # =====================================================================
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_cm = confusion_matrix(y_test, rf_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    rf_report = classification_report(
        y_test, rf_pred, target_names=class_names, output_dict=True
    )

    # =====================================================================
    # SUPPORT VECTOR MACHINE (SVM)
    # =====================================================================
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_clf.fit(X_train, y_train)
    svm_pred = svm_clf.predict(X_test)
    svm_cm = confusion_matrix(y_test, svm_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred)

    svm_report = classification_report(
        y_test, svm_pred, target_names=class_names, output_dict=True
    )

    # =====================================================================
    # KARŞILAŞTIRMALI GÖRSELLEŞTİRME
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Random Forest Confusion Matrix
    sns.heatmap(
        rf_cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0]
    )
    axes[0, 0].set_title(f"Random Forest Confusion Matrix\n(Accuracy: {rf_accuracy:.4f})", 
                         fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Tahmin Edilen", fontsize=11)
    axes[0, 0].set_ylabel("Gerçek Değer", fontsize=11)
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].tick_params(axis="y", rotation=0)

    # SVM Confusion Matrix
    sns.heatmap(
        svm_cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1]
    )
    axes[0, 1].set_title(f"SVM Confusion Matrix\n(Accuracy: {svm_accuracy:.4f})", 
                         fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Tahmin Edilen", fontsize=11)
    axes[0, 1].set_ylabel("Gerçek Değer", fontsize=11)
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].tick_params(axis="y", rotation=0)

    # Random Forest Metrics
    x = np.arange(len(class_names))
    width = 0.25

    rf_precision = [rf_report[c]["precision"] for c in class_names]
    rf_recall = [rf_report[c]["recall"] for c in class_names]
    rf_f1 = [rf_report[c]["f1-score"] for c in class_names]

    axes[1, 0].bar(x - width, rf_precision, width, label="Precision", color="#3498db")
    axes[1, 0].bar(x, rf_recall, width, label="Recall", color="#2ecc71")
    axes[1, 0].bar(x + width, rf_f1, width, label="F1-Score", color="#e74c3c")
    axes[1, 0].set_xlabel("Sınıflar", fontsize=11)
    axes[1, 0].set_ylabel("Skor", fontsize=11)
    axes[1, 0].set_title("Random Forest - Sınıf Bazlı Metrikler", fontsize=14, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    # SVM Metrics
    svm_precision = [svm_report[c]["precision"] for c in class_names]
    svm_recall = [svm_report[c]["recall"] for c in class_names]
    svm_f1 = [svm_report[c]["f1-score"] for c in class_names]

    axes[1, 1].bar(x - width, svm_precision, width, label="Precision", color="#9b59b6")
    axes[1, 1].bar(x, svm_recall, width, label="Recall", color="#f39c12")
    axes[1, 1].bar(x + width, svm_f1, width, label="F1-Score", color="#1abc9c")
    axes[1, 1].set_xlabel("Sınıflar", fontsize=11)
    axes[1, 1].set_ylabel("Skor", fontsize=11)
    axes[1, 1].set_title("SVM - Sınıf Bazlı Metrikler", fontsize=14, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Random Forest vs SVM Karşılaştırması", fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("confusion_matrix_ve_metrikler.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Özet karşılaştırma
    print("\n" + "="*60)
    print("SINIFLANDIRMA YÖNTEMLERİ KARŞILAŞTIRMASI")
    print("="*60)
    print(f"\n1. Random Forest:")
    print(f"   - Accuracy: {rf_accuracy:.4f}")
    print(f"   - Ortalama F1-Score: {np.mean(rf_f1):.4f}")
    
    print(f"\n2. Support Vector Machine (SVM):")
    print(f"   - Accuracy: {svm_accuracy:.4f}")
    print(f"   - Ortalama F1-Score: {np.mean(svm_f1):.4f}")
    
    winner = "Random Forest" if rf_accuracy > svm_accuracy else "SVM"
    print(f"\n>>> En İyi Model: {winner}")
    print("="*60)
