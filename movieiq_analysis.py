"""
MovieIQ — Predictive Analytics on Film Success
Stages 0-4: Data Prep, EDA, Statistical Testing, Modeling

Run:  python movieiq_analysis.py
Outputs:
  - assets/*.png   (all charts)
  - movieiq_model.pkl (trained model, used by the Streamlit app)
  - Printed answers to every question in the brief
"""

import os
import ast
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

# ------------------------------------------------------------------
# STAGE 0 — Problem statement (printed for the record; also in report)
# ------------------------------------------------------------------
print("=" * 70)
print("STAGE 0 — PROBLEM STATEMENT")
print("=" * 70)
print("""
1. A movie is labeled a SUCCESS (1) when its revenue is strictly greater
   than its budget: success = 1 if revenue > budget else 0.
2. Stakeholders:
   - Studios: decide whether to greenlight a script/genre combo and how
     much marketing spend to commit.
   - Investors/production financiers: assess risk before funding a film.
3. Objective: build a model that predicts success from a movie's
   pre-release-ish features (budget, popularity, runtime, vote_average,
   genre) and expose it via an interactive dashboard.
   Steps: (a) clean & label the data, (b) explore relationships via EDA,
   (c) validate signal with statistical tests, (d) train & evaluate a
   Random Forest classifier, (e) ship a Streamlit app for exploration
   and live prediction.
4. Classification problem: the model predicts a discrete category
   (success vs not-success) rather than a continuous number. Target
   variable = `success` (0/1).
""")

# ------------------------------------------------------------------
# STAGE 1 — Data Preparation
# ------------------------------------------------------------------
print("=" * 70)
print("STAGE 1 — DATA PREPARATION")
print("=" * 70)

df = pd.read_csv("movies.csv")

print(f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nSummary statistics (numeric columns):")
print(df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())

# Missing values / zeros check
print("\nMissing values per column:")
print(df.isnull().sum())

zero_budget = (df["budget"] == 0).sum()
zero_revenue = (df["revenue"] == 0).sum()
print(f"\nRows with budget == 0: {zero_budget}")
print(f"Rows with revenue == 0: {zero_revenue}")
print("""
A budget or revenue of 0 usually means the value was never recorded
(not that the movie truly cost/made nothing), so it's missing data in
disguise. Since success = revenue > budget, a genuine zero would
silently mislabel the row. The fix: drop rows where either field is 0
(there are none in this dataset, but the check is required for any new
data pulled from TMDB-style sources).
""")
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

# Target column
df["success"] = (df["revenue"] > df["budget"]).astype(int)
success_rate = df["success"].mean()
print(f"Proportion of successful movies: {success_rate:.2%}")
print("Balanced" if 0.4 <= success_rate <= 0.6 else "Not perfectly balanced (still workable)")

# Genre parsing: "[{'id': 18, 'name': 'Drama'}]" -> ['Drama']
def parse_genres(raw):
    try:
        items = ast.literal_eval(raw)
        return [d["name"] for d in items]
    except (ValueError, SyntaxError):
        return []

df["genre_list"] = df["genres"].apply(parse_genres)
df["primary_genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")

print("\nSample of parsed genres:")
print(df[["title", "genres", "genre_list", "primary_genre"]].head(3))

# ------------------------------------------------------------------
# STAGE 2 — Exploratory Data Analysis
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 1. Budget vs Revenue scatter
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="budget", y="revenue", hue="success", alpha=0.6, palette={0: "crimson", 1: "seagreen"})
plt.title("Budget vs Revenue")
plt.xlabel("Budget ($)")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/budget_vs_revenue.png", dpi=150)
plt.close()

corr_br = df["budget"].corr(df["revenue"])
print(f"\nBudget-Revenue correlation: {corr_br:.3f}")
print("Higher budgets show a positive but noisy relationship with revenue —"
      " there's a general upward trend, but plenty of high-budget films"
      " still fail to out-earn their budget.")

# 2. Genre trends
genre_df = df.explode("genre_list")
genre_counts = genre_df["genre_list"].value_counts()
genre_success = genre_df.groupby("genre_list")["success"].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
genre_counts.head(10).plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Most Common Genres")
axes[0].set_ylabel("Count")
genre_success.head(10).plot(kind="bar", ax=axes[1], color="seagreen")
axes[1].set_title("Highest Success-Rate Genres")
axes[1].set_ylabel("Success Rate")
plt.tight_layout()
plt.savefig(f"{ASSETS}/genre_trends.png", dpi=150)
plt.close()

print("\nTop 5 most common genres:")
print(genre_counts.head(5))
print("\nTop 5 genres by success rate:")
print(genre_success.head(5))

# 3. Popularity, runtime, vote_average vs success
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
    sns.boxplot(data=df, x="success", y=col, hue="success", ax=ax,
                palette={0: "crimson", 1: "seagreen"}, legend=False)
    ax.set_title(f"{col} by success")
    ax.set_xticklabels(["Fail", "Success"])
plt.tight_layout()
plt.savefig(f"{ASSETS}/feature_vs_success_boxplots.png", dpi=150)
plt.close()

means_by_success = df.groupby("success")[["popularity", "runtime", "vote_average"]].mean()
print("\nMean feature values by success:")
print(means_by_success)

# 4. Correlation heatmap
plt.figure(figsize=(7, 6))
num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{ASSETS}/correlation_heatmap.png", dpi=150)
plt.close()

print("\nCorrelation matrix:")
print(corr_matrix.round(2))
print("""
Note: budget/revenue are strongly related to `success` by construction
(success is literally derived from them), so they are excluded from
the model's input features to avoid leakage — see Stage 4.
""")

# ------------------------------------------------------------------
# STAGE 3 — Statistical Testing
# ------------------------------------------------------------------
print("=" * 70)
print("STAGE 3 — STATISTICAL TESTING")
print("=" * 70)

# T-test: vote_average, successful vs unsuccessful
success_votes = df.loc[df["success"] == 1, "vote_average"]
fail_votes = df.loc[df["success"] == 0, "vote_average"]
t_stat, p_val_t = stats.ttest_ind(success_votes, fail_votes, equal_var=False)

print(f"""
T-TEST — vote_average, successful vs unsuccessful movies
  H0: mean vote_average is equal between successful and unsuccessful movies.
  t-statistic = {t_stat:.3f}, p-value = {p_val_t:.4f}
  Conclusion: {"Reject H0 — vote_average differs significantly (p < 0.05)"
               if p_val_t < 0.05 else
               "Fail to reject H0 — no significant difference found (p >= 0.05)"}
""")

# Chi-square: primary_genre vs success
contingency = pd.crosstab(df["primary_genre"], df["success"])
chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency)

print(f"""
CHI-SQUARE TEST — primary_genre vs success
  H0: genre and success are independent (no association).
  chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_val_chi:.4f}
  Conclusion: {"Reject H0 — genre is significantly associated with success (p < 0.05)"
               if p_val_chi < 0.05 else
               "Fail to reject H0 — no significant association found (p >= 0.05)"}
""")

print("""
What a p-value means (plain language): it's the probability of seeing
a difference at least as extreme as the one observed, IF the null
hypothesis (no real effect) were actually true. A small p-value means
the observed pattern would be unlikely under "no effect," so we treat
it as evidence of a real effect. Threshold used: 0.05 (standard
convention balancing false positives vs false negatives).
""")

# ------------------------------------------------------------------
# STAGE 4 — Predictive Modeling (Random Forest)
# ------------------------------------------------------------------
print("=" * 70)
print("STAGE 4 — PREDICTIVE MODELING")
print("=" * 70)

# Features: exclude title (identifier, no predictive value) and revenue
# (it's used to DEFINE the label -> including it would be data leakage).
feature_cols = ["budget", "popularity", "runtime", "vote_average"]
df_model = pd.get_dummies(df, columns=["primary_genre"], prefix="genre_cat")
genre_dummy_cols = [c for c in df_model.columns if c.startswith("genre_cat_")]
feature_cols_full = feature_cols + genre_dummy_cols

X = df_model[feature_cols_full]
y = df_model["success"]

print(f"\nFeatures used ({len(feature_cols_full)}): {feature_cols} + one-hot genre columns")
print("Excluded: `title` (identifier only), `revenue` (used to derive the label — leakage).")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain/test split: 80/20 ({len(X_train)} train, {len(X_test)} test)")
print("A held-out test set is necessary to estimate how the model performs"
      " on movies it has never seen, avoiding overly optimistic accuracy"
      " from just re-scoring the training data.")

rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

print("""
How a Random Forest predicts: it builds many decision trees, each
trained on a random subset of rows and features. Each tree votes
"success" or "not success" based on a series of yes/no splits on
feature values (e.g. "is budget < $50M?"). The forest's final
prediction is the majority vote across all trees, which smooths out
the mistakes any single tree would make.
""")

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print("\nConfusion matrix:")
print(cm)
print("\nFull classification report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Success"], yticklabels=["Fail", "Success"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{ASSETS}/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols_full).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
importances.head(10).plot(kind="barh", color="darkorange")
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{ASSETS}/feature_importance.png", dpi=150)
plt.close()

print("\nTop feature importances:")
print(importances.head(10))

# Save model + metadata for the Streamlit app
joblib.dump(
    {
        "model": rf,
        "feature_cols": feature_cols,
        "genre_dummy_cols": genre_dummy_cols,
        "genres_known": [c.replace("genre_cat_", "") for c in genre_dummy_cols],
    },
    "movieiq_model.pkl",
)

print("\nModel saved to movieiq_model.pkl")
print("\nAll charts saved to assets/")
print("Done.")
