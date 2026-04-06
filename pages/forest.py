import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

def render():
    st.markdown('<div class="section-title" style="margin-top:0">🌲 Random Forest & Ensemble Methods</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Concept", "🎬 Forest vs Single Tree", "🧪 Feature Importance"])

    with tab1:
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            <div class="algo-card">
                <div class="algo-card-accent" style="background:#1A56DB;"></div>
                <div class="algo-card-title" style="color:#1A56DB; margin-left:1rem;">What is Random Forest?</div>
                <div class="algo-card-desc" style="margin-left:1rem; font-size:0.93rem; color:#E2EAF4; margin-top:0.5rem;">
                    Random Forest is an <strong>ensemble learning</strong> method that builds many 
                    Decision Trees on random subsets of data (<strong>bootstrap sampling</strong>) and 
                    random subsets of features. The final prediction is made by <strong>majority vote</strong> 
                    (classification) or <strong>averaging</strong> (regression).
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="formula-box">ŷ = mode { Tree₁(x), Tree₂(x), ..., Treeₙ(x) }</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#0A1220; border:1px solid #1A56DB; border-radius:8px; padding:1rem; margin-top:0.8rem; font-size:0.88rem;">
                <strong style="color:#00C2FF;">Why does it work better?</strong><br>
                <span style="color:#E2EAF4;">Each tree is trained on different data (bootstrap) and uses only a random 
                subset of features at each split. This <strong>decorrelates</strong> the trees — their errors don't 
                all point in the same direction, so averaging reduces variance and prevents overfitting.</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-title" style="font-size:1rem;">How It Works</div>', unsafe_allow_html=True)
            steps = [
                ("Bootstrap Sampling", "Create N random subsets of the training data (with replacement)."),
                ("Build Trees", "Train one Decision Tree on each bootstrap sample."),
                ("Random Features", "At each node, consider only √d random features for splitting."),
                ("Make Predictions", "Each tree independently predicts the class for new data."),
                ("Majority Vote", "Combine all tree predictions via majority vote (classification)."),
            ]
            for i, (title, desc) in enumerate(steps):
                st.markdown(f"""
                <div class="step-box">
                    <div class="step-num">{i+1}</div>
                    <div class="step-text"><strong style="color:#fff;">{title}:</strong> {desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("Compare a Single Decision Tree vs Random Forest accuracy as you vary tree depth and forest size.")
        col1, col2, col3 = st.columns(3)
        with col1: n_trees = st.slider("Number of Trees", 1, 200, 50)
        with col2: max_d = st.slider("Max Depth (Single Tree)", 1, 15, 5)
        with col3: n_samples = st.slider("Dataset Size", 100, 500, 200)

        np.random.seed(42)
        X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5,
                                    random_state=42, n_redundant=2)
        from sklearn.model_selection import cross_val_score

        single = DecisionTreeClassifier(max_depth=max_d, random_state=0)
        forest = RandomForestClassifier(n_estimators=n_trees, max_depth=max_d, random_state=0)

        sc_single = cross_val_score(single, X, y, cv=5).mean()
        sc_forest = cross_val_score(forest, X, y, cv=5).mean()

        fig = go.Figure(go.Bar(
            x=['Single Decision Tree', 'Random Forest'],
            y=[sc_single, sc_forest],
            marker_color=['#F59E0B', '#1A56DB'],
            text=[f'{sc_single:.2%}', f'{sc_forest:.2%}'],
            textposition='outside',
            textfont=dict(color='#E2EAF4', size=14),
        ))
        fig.update_layout(
            paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'),
            yaxis=dict(title='5-Fold CV Accuracy', range=[0, 1.05], gridcolor='#1E3A5F'),
            title=dict(text='Single Tree vs Random Forest (5-fold CV)', font=dict(color='#E2EAF4')),
            height=350, margin=dict(t=50, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        improvement = (sc_forest - sc_single) * 100
        if improvement > 0:
            st.success(f"🌲 Random Forest is **+{improvement:.1f}%** more accurate than a single tree!")
        else:
            st.info(f"Single tree matches forest here — try adjusting parameters.")

        # Accuracy vs n_estimators
        st.markdown('<div class="section-title" style="font-size:1rem;">Accuracy vs Number of Trees</div>', unsafe_allow_html=True)
        tree_counts = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
        accs = [cross_val_score(RandomForestClassifier(n_estimators=n, random_state=0), X, y, cv=3).mean() for n in tree_counts]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tree_counts, y=accs, mode='lines+markers',
            line=dict(color='#1A56DB', width=2.5),
            marker=dict(color='#00C2FF', size=8, line=dict(color='white', width=1))))
        fig2.add_hline(y=sc_single, line_dash='dash', line_color='#F59E0B',
                       annotation_text=f"Single Tree: {sc_single:.2%}")
        fig2.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'),
            xaxis=dict(title='Number of Trees', gridcolor='#1E3A5F'),
            yaxis=dict(title='CV Accuracy', gridcolor='#1E3A5F'),
            height=280, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("See which features the Random Forest considers most important.")
        col1, col2 = st.columns(2)
        with col1: n_trees3 = st.slider("Trees", 10, 200, 100, key="fi_trees")
        with col2: n_feats = st.slider("Features", 4, 15, 8, key="fi_feats")

        np.random.seed(0)
        X3, y3 = make_classification(n_samples=300, n_features=n_feats, n_informative=min(5, n_feats),
                                      n_redundant=min(2, n_feats-3), random_state=0)
        rf = RandomForestClassifier(n_estimators=n_trees3, random_state=0)
        rf.fit(X3, y3)
        importances = rf.feature_importances_
        feat_names = [f"Feature {i+1}" for i in range(n_feats)]
        sorted_idx = np.argsort(importances)[::-1]

        fig3 = go.Figure(go.Bar(
            x=[feat_names[i] for i in sorted_idx],
            y=[importances[i] for i in sorted_idx],
            marker=dict(
                color=[importances[i] for i in sorted_idx],
                colorscale=[[0,'#1A3A5C'],[1,'#1A56DB']],
                showscale=False,
                line=dict(color='rgba(0,0,0,0)')
            ),
            text=[f'{importances[i]:.3f}' for i in sorted_idx],
            textposition='outside',
            textfont=dict(color='#E2EAF4', size=11),
        ))
        fig3.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'),
            xaxis=dict(gridcolor='#1E3A5F', title='Features'),
            yaxis=dict(gridcolor='#1E3A5F', title='Importance (Gini)'),
            title=dict(text='Feature Importance (Mean Decrease in Impurity)', font=dict(color='#E2EAF4')),
            height=380, margin=dict(t=50))
        st.plotly_chart(fig3, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("RF Accuracy", f"{rf.score(X3,y3):.2%}")
        c2.metric("Top Feature", feat_names[sorted_idx[0]])
