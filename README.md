# 🧠 Animated Presentation on Machine Learning Algorithms
### A Streamlit Web App

---

## 📁 Project Structure

```
ml_project/
│
├── app.py                  ← Main Streamlit app (run this)
├── requirements.txt        ← Python dependencies
├── setup_and_run.bat       ← Windows one-click setup & run
│
└── pages/
    ├── home.py             ← Home page with overview & radar chart
    ├── linear.py           ← Linear Regression
    ├── decision.py         ← Decision Trees
    ├── kmeans.py           ← K-Means Clustering
    ├── svm.py              ← Support Vector Machine
    ├── neural.py           ← Neural Networks
    ├── forest.py           ← Random Forest
    └── compare.py          ← Algorithm Comparison & Benchmark
```

---

## 🚀 Quick Start (Windows)

### Option 1 — One-Click (Recommended)
Double-click `setup_and_run.bat`

That's it! It will:
1. Create a virtual environment
2. Install all dependencies
3. Launch the app at http://localhost:8501

---

### Option 2 — Manual Steps

**Step 1: Open Command Prompt in this folder**
```
Right-click inside the ml_project folder → "Open in Terminal"
```

**Step 2: Create & activate virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Step 3: Install requirements**
```cmd
pip install -r requirements.txt
```

**Step 4: Run the app**
```cmd
streamlit run app.py
```

**Step 5: Open browser**
Go to: http://localhost:8501

---

## 📚 Algorithms Covered

| # | Algorithm | Type | Page |
|---|-----------|------|------|
| 1 | Linear Regression | Supervised | `pages/linear.py` |
| 2 | Decision Trees | Supervised | `pages/decision.py` |
| 3 | K-Means Clustering | Unsupervised | `pages/kmeans.py` |
| 4 | Support Vector Machine | Supervised | `pages/svm.py` |
| 5 | Neural Networks | Deep Learning | `pages/neural.py` |
| 6 | Random Forest | Ensemble | `pages/forest.py` |

---

## 🛠 Requirements

- Python 3.9 or higher
- Windows 10 / 11
- Internet connection (first run only, to download packages)

---

## 🎓 Features

- **📖 Concept** tab — Theory, formulas, use cases
- **🎬 Animated Demo** tab — Interactive slider-based visualization
- **🧪 Live Experiment** tab — Tweak parameters and see results instantly
- **📊 Comparison page** — Benchmark all algorithms on one dataset

---

Built with ❤️ using Streamlit, Plotly, and Scikit-learn
