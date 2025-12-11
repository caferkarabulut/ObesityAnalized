import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_regression(csv_path: str) -> None:
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

    # Avoid leakage: drop BMI target, label, and Weight/Height used in BMI formula
    X_reg = df.drop(["BMI", "NObeyesdad", "Weight", "Height"], axis=1)
    y_reg = df["BMI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)

    # Plot: True vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred_lr, alpha=0.5, color="#3498db", edgecolors="white", s=50)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 "r--", lw=2, label="Mükemmel Tahmin (y=x)")
    axes[0].set_xlabel("Gerçek BMI", fontsize=12)
    axes[0].set_ylabel("Tahmin BMI", fontsize=12)
    axes[0].set_title(f"Linear Regression\nR² = {lr_r2:.4f}, RMSE = {lr_rmse:.4f}",
                      fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test, y_pred_rf, alpha=0.5, color="#2ecc71", edgecolors="white", s=50)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 "r--", lw=2, label="Mükemmel Tahmin (y=x)")
    axes[1].set_xlabel("Gerçek BMI", fontsize=12)
    axes[1].set_ylabel("Tahmin BMI", fontsize=12)
    axes[1].set_title(f"Random Forest Regressor\nR² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}",
                      fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Gerçek BMI vs Tahmin BMI Karşılaştırması", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("bmi_regression_sonuclari.png", dpi=150, bbox_inches="tight")
    plt.show()
