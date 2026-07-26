# 🎬 MovieIQ — Predictive Analytics on Film Success

**Live app:** https://movieiqproject-dsjhvpepnjfdsjbbrvucny.streamlit.app/

MovieIQ is an interactive dashboard that explores and predicts movie success from budget, revenue, popularity, runtime, and audience ratings. A movie is labeled **successful** when its revenue exceeds its budget. The project covers the full pipeline: data cleaning, exploratory analysis, statistical hypothesis testing, a Random Forest classifier, and a deployed Streamlit app for live prediction.

---

## Features

- **Filters** — narrow the dataset by genre and minimum vote average via the sidebar
- **EDA charts** — budget vs. revenue scatter, genre success rates, feature distributions by outcome, correlation heatmap
- **Statistical tests** — live T-test (vote average) and Chi-square test (genre) results on the filtered data
- **Live prediction** — enter a movie's budget, popularity, runtime, vote average, and genre to get a success/failure prediction with confidence score

## Tech Stack

| Purpose | Library |
|---|---|
| Data handling | Pandas, NumPy |
| Modeling | scikit-learn (Random Forest) |
| Statistics | SciPy |
| Visualization | Matplotlib, Seaborn |
| App / Dashboard | Streamlit |

## Project Structure

```
MovieIQ.py              # Streamlit dashboard (deployed app entry point)
movieiq_analysis.py      # Full analysis pipeline: EDA, stats tests, model training
movies.csv                # Dataset (budget, revenue, popularity, runtime, vote_average, title, genres)
movieiq_model.pkl         # Trained Random Forest model, used by the dashboard
requirements.txt          # Python dependencies
assets/                   # Saved chart images from the analysis script
```

## Run Locally

```bash
pip install -r requirements.txt
python movieiq_analysis.py   # generates charts + trains/saves the model
streamlit run MovieIQ.py     # launches the dashboard at localhost:8501
```

## Model Notes

- **Target:** `success` = 1 if `revenue > budget`, else 0
- **Features used:** budget, popularity, runtime, vote_average, and one-hot encoded genre
- **Excluded:** `title` (identifier only) and `revenue` (used to define the label — including it would leak the answer into the model)
- **Performance:** ~74% accuracy, 81% precision / 90% recall on the success class

- Jainoor Singh Saini B.Tech Student & Business Analytics Intern
