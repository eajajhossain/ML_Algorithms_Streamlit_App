import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def render():
    st.markdown('<div class="section-title" style="margin-top:0">📊 Algorithm Comparison</div>', unsafe_allow_html=True)

    # Static comparison table
    st.markdown("""
    <table class="cmp-table">
        <tr>
            <th>Algorithm</th><th>Type</th><th>Interpretable</th>
            <th>Scalable</th><th>Non-Linear</th><th>Training Speed</th><th>Best For</th>
        </tr>
        <tr><td>Linear Regression</td><td>Supervised</td>
            <td class="badge-yes">●●●</td><td class="badge-yes">●●●</td>
            <td class="badge-no">✗</td><td class="badge-yes">Fast</td><td>Continuous prediction</td></tr>
        <tr><td>Decision Tree</td><td>Supervised</td>
            <td class="badge-yes">●●●</td><td class="badge-med">●●</td>
            <td class="badge-med">●●</td><td class="badge-yes">Fast</td><td>Explainable classification</td></tr>
        <tr><td>K-Means</td><td>Unsupervised</td>
            <td class="badge-med">●●</td><td class="badge-yes">●●●</td>
            <td class="badge-med">●</td><td class="badge-yes">Fast</td><td>Customer segmentation</td></tr>
        <tr><td>SVM</td><td>Supervised</td>
            <td class="badge-no">●</td><td class="badge-no">●</td>
            <td class="badge-yes">●●●</td><td class="badge-med">Medium</td><td>High-dimensional data</td></tr>
        <tr><td>Neural Network</td><td>Deep Learning</td>
            <td class="badge-no">✗</td><td class="badge-yes">●●●</td>
            <td class="badge-yes">●●●</td><td class="badge-no">Slow</td><td>Complex pattern recognition</td></tr>
        <tr><td>Random Forest</td><td>Ensemble</td>
            <td class="badge-med">●●</td><td class="badge-med">●●</td>
            <td class="badge-yes">●●●</td><td class="badge-med">Medium</td><td>General-purpose accuracy</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Live benchmark
    st.markdown('<div class="section-title">🏆 Live Accuracy Benchmark</div>', unsafe_allow_html=True)
    st.markdown("Run all classifiers on the same dataset and compare their 5-fold cross-validated accuracy.")

    col1, col2 = st.columns(2)
    with col1: n_samples = st.slider("Dataset Size", 100, 800, 300)
    with col2: n_features = st.slider("Number of Features", 4, 20, 10)

    if st.button("▶  Run Benchmark", type="primary"):
        np.random.seed(42)
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                    n_informative=min(6,n_features), n_redundant=2, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            "Linear (LR)": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "SVM (RBF)": SVC(kernel='rbf', gamma='scale'),
            "Neural Net": MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=0),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
        }
        colors = ['#10B981','#F59E0B','#00C2FF','#EF4444','#1A56DB']

        results = {}
        prog = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            scores = cross_val_score(model, X_scaled, y, cv=5)
            results[name] = (scores.mean(), scores.std())
            prog.progress((i+1)/len(models))

        names = list(results.keys())
        means = [results[n][0] for n in names]
        stds  = [results[n][1] for n in names]
        sorted_idx = np.argsort(means)[::-1]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[names[i] for i in sorted_idx],
            y=[means[i] for i in sorted_idx],
            error_y=dict(type='data', array=[stds[i] for i in sorted_idx], visible=True,
                         color='rgba(255,255,255,0.5)', thickness=1.5, width=4),
            marker_color=[colors[i] for i in sorted_idx],
            text=[f'{means[i]:.2%}' for i in sorted_idx],
            textposition='outside',
            textfont=dict(color='#E2EAF4', size=12),
        ))
        fig.update_layout(
            paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'),
            yaxis=dict(title='5-Fold CV Accuracy', range=[0, 1.1], gridcolor='#1E3A5F'),
            xaxis=dict(gridcolor='#1E3A5F'),
            title=dict(text='Algorithm Benchmark (5-Fold Cross-Validation)', font=dict(color='#E2EAF4')),
            height=400, margin=dict(t=50, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        best = names[sorted_idx[0]]
        st.success(f"🥇 **Best algorithm for this dataset:** {best} ({means[sorted_idx[0]]:.2%} accuracy)")

        # Radar chart
        st.markdown('<div class="section-title">🕸 Multi-Dimension Radar</div>', unsafe_allow_html=True)
        cats = ['Accuracy', 'Speed', 'Interpretability', 'Scalability', 'Non-Linear']
        scores_static = {
            "Linear (LR)":    [means[names.index("Linear (LR)")]*10, 9, 10, 9, 2],
            "Decision Tree":  [means[names.index("Decision Tree")]*10, 8, 9, 7, 7],
            "SVM (RBF)":      [means[names.index("SVM (RBF)")]*10, 4, 4, 5, 9],
            "Neural Net":     [means[names.index("Neural Net")]*10, 5, 2, 9, 10],
            "Random Forest":  [means[names.index("Random Forest")]*10, 7, 6, 8, 9],
        }
        fig2 = go.Figure()
        for (name, vals), color in zip(scores_static.items(), colors):
            fig2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill='toself', name=name,
                line_color=color, fillcolor=color, opacity=0.18))
        fig2.update_layout(
            polar=dict(bgcolor='#0A1220',
                radialaxis=dict(visible=True, range=[0,10], color='#7A9CC0', gridcolor='#1E3A5F'),
                angularaxis=dict(color='#E2EAF4', gridcolor='#1E3A5F')),
            paper_bgcolor='#162032', plot_bgcolor='#162032',
            font=dict(color='#E2EAF4'),
            legend=dict(bgcolor='#0A1220', bordercolor='#1E3A5F'),
            height=400, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#0A1220; border:1px solid #1E3A5F; border-radius:10px; padding:2rem; text-align:center; color:#7A9CC0;">
            ▶ Click <strong style="color:#00C2FF;">Run Benchmark</strong> to train all algorithms and compare results live!
        </div>
        """, unsafe_allow_html=True)
