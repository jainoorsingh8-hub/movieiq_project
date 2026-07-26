"""
MovieIQ — Streamlit Dashboard
Run:  streamlit run MovieIQ.py
Requires movies.csv and movieiq_model.pkl (produced by movieiq_analysis.py)
in the same folder.
"""

import ast
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

st.set_page_config(page_title="MovieIQ", layout="wide")


# ---------- Data loading & caching ----------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    def parse_genres(raw):
        try:
            return [d["name"] for d in ast.literal_eval(raw)]
        except (ValueError, SyntaxError):
            return []

    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")
    return df


@st.cache_resource
def load_model():
    return joblib.load("movieiq_model.pkl")


df = load_data()
bundle = load_model()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
genre_dummy_cols = bundle["genre_dummy_cols"]
genres_known = bundle["genres_known"]

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
all_genres = sorted(df["primary_genre"].unique())
selected_genres = st.sidebar.multiselect("Genre", all_genres, default=all_genres)
min_vote = st.sidebar.slider(
    "Minimum vote average", float(df["vote_average"].min()), float(df["vote_average"].max()), float(df["vote_average"].min())
)

filtered = df[df["primary_genre"].isin(selected_genres) & (df["vote_average"] >= min_vote)]

st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption(f"Showing {len(filtered)} of {len(df)} movies after filters")

# ---------- KPI row ----------
col1, col2, col3 = st.columns(3)
col1.metric("Movies (filtered)", len(filtered))
col2.metric("Success rate", f"{filtered['success'].mean():.1%}" if len(filtered) else "N/A")
col3.metric("Avg. budget", f"${filtered['budget'].mean():,.0f}" if len(filtered) else "N/A")

st.divider()

# ---------- EDA charts ----------
st.header("Exploratory Data Analysis")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Budget vs Revenue")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered, x="budget", y="revenue", hue="success",
                     palette={0: "crimson", 1: "seagreen"}, alpha=0.6, ax=ax, legend=True)
    st.pyplot(fig)

with c2:
    st.subheader("Success Rate by Genre")
    genre_success = (
        filtered.explode("genre_list").groupby("genre_list")["success"].mean().sort_values(ascending=False)
    )
    fig, ax = plt.subplots()
    genre_success.plot(kind="bar", ax=ax, color="seagreen")
    ax.set_ylabel("Success Rate")
    st.pyplot(fig)

c3, c4 = st.columns(2)

with c3:
    st.subheader("Feature Distributions by Success")
    feat = st.selectbox("Feature", ["popularity", "runtime", "vote_average"])
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered, x="success", y=feat, hue="success",
                palette={0: "crimson", 1: "seagreen"}, legend=False, ax=ax)
    ax.set_xticklabels(["Fail", "Success"])
    st.pyplot(fig)

with c4:
    st.subheader("Correlation Heatmap")
    num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    fig, ax = plt.subplots()
    sns.heatmap(filtered[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

st.divider()

# ---------- Statistical test results ----------
st.header("Statistical Test Results")

t1, t2 = st.columns(2)
with t1:
    st.subheader("T-Test: vote_average vs success")
    if filtered["success"].nunique() == 2:
        s = filtered.loc[filtered["success"] == 1, "vote_average"]
        f = filtered.loc[filtered["success"] == 0, "vote_average"]
        t_stat, p_val = stats.ttest_ind(s, f, equal_var=False)
        st.write(f"t-statistic = `{t_stat:.3f}`, p-value = `{p_val:.4f}`")
        st.write("Significant difference" if p_val < 0.05 else "No significant difference")
    else:
        st.info("Need both success and failure rows in the current filter to run this test.")

with t2:
    st.subheader("Chi-Square: genre vs success")
    if filtered["success"].nunique() == 2 and filtered["primary_genre"].nunique() > 1:
        contingency = pd.crosstab(filtered["primary_genre"], filtered["success"])
        chi2, p_val_chi, dof, _ = stats.chi2_contingency(contingency)
        st.write(f"chi2 = `{chi2:.3f}`, dof = `{dof}`, p-value = `{p_val_chi:.4f}`")
        st.write("Significant association" if p_val_chi < 0.05 else "No significant association")
    else:
        st.info("Need multiple genres and both classes present to run this test.")

st.divider()

# ---------- Prediction section ----------
st.header("Predict Success for a New Movie")

with st.form("predict_form"):
    p1, p2 = st.columns(2)
    with p1:
        in_budget = st.number_input("Budget ($)", min_value=1000, value=50_000_000, step=1_000_000)
        in_popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
    with p2:
        in_runtime = st.number_input("Runtime (minutes)", min_value=1, value=120)
        in_vote = st.slider("Vote average", 0.0, 10.0, 6.0)
    in_genre = st.selectbox("Primary genre", genres_known)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {c: 0 for c in genre_dummy_cols}
    genre_col = f"genre_cat_{in_genre}"
    if genre_col in row:
        row[genre_col] = 1
    row.update(
        {"budget": in_budget, "popularity": in_popularity, "runtime": in_runtime, "vote_average": in_vote}
    )
    X_new = pd.DataFrame([row])[feature_cols + genre_dummy_cols]
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]

    if pred == 1:
        st.success(f"Predicted: SUCCESS ✅ (confidence: {proba:.1%})")
    else:
        st.error(f"Predicted: NOT SUCCESSFUL ❌ (confidence: {1 - proba:.1%})")

st.caption(
    "Reflection: this model relies on features like budget and popularity that are "
    "themselves influenced by marketing spend and buzz, not just raw movie quality — "
    "treat predictions as directional signal, not certainty."
)
