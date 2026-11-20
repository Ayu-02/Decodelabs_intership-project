import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)



df = pd.read_csv("dataset/creditcard.csv")

print("Dataset Loaded Successfully!")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()



print("\nMissing Values:")
print(df.isnull().sum().sum())


df = df.drop_duplicates()


df = df.dropna(subset=["Class"])


X = df.drop("Class", axis=1)
y = df["Class"]


y = y.astype(int)

print("\nCleaned Dataset Shape:", df.shape)



print("\nClass Distribution:")
print(y.value_counts())

print("\nFraud Percentage:")
print(y.value_counts(normalize=True) * 100)

plt.figure(figsize=(7, 5))
sns.countplot(x=y)
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.savefig("outputs/class_distribution.png")
plt.show()



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)



logistic_pipeline = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])



rf_pipeline = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])



print("\nTraining Logistic Regression...")

logistic_pipeline.fit(X_train, y_train)

print("Logistic Regression Training Completed!")




print("\nTraining Random Forest...")

rf_pipeline.fit(X_train, y_train)

print("Random Forest Training Completed!")




print("\nStarting Hyperparameter Tuning...")

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_grid,
    n_iter=5,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

best_rf_model = random_search.best_estimator_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(random_search.best_score_)




models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": rf_pipeline,
    "Tuned Random Forest": best_rf_model
}

results = []

for name, model in models.items():

    
    y_pred = model.predict(X_test)

    
    y_prob = model.predict_proba(X_test)[:, 1]

    
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    pr_auc = average_precision_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    print("\n================================")
    print("MODEL:", name)
    print("================================")

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))



results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

best_model_name = results_df.loc[
    results_df["ROC-AUC"].idxmax(), "Model"
]

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)




best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.show()




plt.figure(figsize=(8, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(y_test, probabilities)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.savefig("outputs/roc_curve.png")
plt.show()




results_plot = results_df.set_index("Model")[
    ["Precision", "Recall", "ROC-AUC"]
]

results_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("outputs/model_comparison.png")
plt.show()



joblib.dump(
    best_model,
    "models/best_fraud_model.pkl"
)

# =========================================
# DISPLAY ALL GRAPHS IN ONE WINDOW
# =========================================

print("\nGenerating All Graphs...")

# Create one figure with 4 graphs
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# -----------------------------------------
# GRAPH 1: CLASS DISTRIBUTION
# -----------------------------------------

sns.countplot(x=y, ax=axes[0, 0])

axes[0, 0].set_title("Legitimate vs Fraud Transactions")
axes[0, 0].set_xlabel("Transaction Class")
axes[0, 0].set_ylabel("Number of Transactions")


# -----------------------------------------
# GRAPH 2: CONFUSION MATRIX
# -----------------------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
    ax=axes[0, 1]
)

axes[0, 1].set_title("Confusion Matrix")
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("Actual")


# -----------------------------------------
# GRAPH 3: ROC CURVE
# -----------------------------------------

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(y_test, probabilities)

    axes[1, 0].plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_value:.3f})"
    )

axes[1, 0].plot([0, 1], [0, 1], "--")

axes[1, 0].set_title("ROC Curve Comparison")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend()


# -----------------------------------------
# GRAPH 4: MODEL COMPARISON
# -----------------------------------------

results_plot = results_df.set_index("Model")[
    ["Precision", "Recall", "ROC-AUC"]
]

results_plot.plot(
    kind="bar",
    ax=axes[1, 1]
)

axes[1, 1].set_title("Model Performance Comparison")
axes[1, 1].set_ylabel("Score")
axes[1, 1].set_ylim(0, 1.05)
axes[1, 1].tick_params(axis="x", rotation=20)
axes[1, 1].legend(loc="lower right")


# -----------------------------------------
# SAVE ALL GRAPHS
# -----------------------------------------

plt.tight_layout()

plt.savefig(
    "outputs/all_graphs.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nAll Graphs Saved Successfully!")
print("Location: outputs/all_graphs.png")

# Display graphs
plt.show()

plt.close()


print("\nBest Model Saved Successfully!")



print("\n================================")
print("FRAUD DETECTION PROJECT COMPLETED")
print("================================")

print("Selected Model:", best_model_name)

print("\nGenerated Files:")
print("1. outputs/class_distribution.png")
print("2. outputs/confusion_matrix.png")
print("3. outputs/roc_curve.png")
print("4. outputs/model_comparison.png")
print("5. outputs/model_comparison.csv")
print("6. models/best_fraud_model.pkl")
