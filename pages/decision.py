import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

def render():
    st.markdown('<div class="section-title" style="margin-top:0">🌳 Decision Trees</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Concept", "🎬 Decision Boundary", "🧪 Live Experiment"])

    with tab1:
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            <div class="algo-card">
                <div class="algo-card-accent" style="background:#F59E0B;"></div>
                <div class="algo-card-title" style="color:#F59E0B; margin-left:1rem;">What is a Decision Tree?</div>
                <div class="algo-card-desc" style="margin-left:1rem; font-size:0.93rem; color:#E2EAF4; margin-top:0.5rem;">
                    A Decision Tree is a flowchart-like structure where each <strong>internal node</strong> represents 
                    a feature test, each <strong>branch</strong> is the outcome of that test, and each <strong>leaf node</strong> 
                    holds the final prediction. The tree is built by recursively choosing the best feature split.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Gini = 1 − Σ pᵢ²</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Entropy = −Σ pᵢ · log₂(pᵢ)</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Info Gain = Entropy(parent) − Weighted Entropy(children)</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title" style="font-size:1rem;">Building the Tree</div>', unsafe_allow_html=True)
            steps = [
                ("Start at Root", "Use the entire training dataset at the root node."),
                ("Select Best Split", "Evaluate every feature & threshold using Gini or Entropy."),
                ("Split the Node", "Partition data into left (True) and right (False) child nodes."),
                ("Recurse", "Repeat steps 2–3 on each child node recursively."),
                ("Stop & Label", "Stop when max depth reached or node is pure; assign class label."),
            ]
            for i, (title, desc) in enumerate(steps):
                st.markdown(f"""
                <div class="step-box">
                    <div class="step-num">{i+1}</div>
                    <div class="step-text"><strong style="color:#fff;">{title}:</strong> {desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Pros & Cons</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="background:#0A2010; border:1px solid #10B981; border-radius:8px; padding:1rem;">
                <div style="color:#10B981; font-weight:700; margin-bottom:0.5rem;">✅ Advantages</div>
                <div style="font-size:0.88rem; color:#E2EAF4; line-height:1.8;">
                • Easy to understand & visualize<br>
                • Handles both numerical & categorical<br>
                • No feature scaling needed<br>
                • Fast training & prediction
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="background:#200A0A; border:1px solid #EF4444; border-radius:8px; padding:1rem;">
                <div style="color:#EF4444; font-weight:700; margin-bottom:0.5rem;">❌ Disadvantages</div>
                <div style="font-size:0.88rem; color:#E2EAF4; line-height:1.8;">
                • Prone to overfitting<br>
                • Unstable (small data change → new tree)<br>
                • Biased with imbalanced datasets<br>
                • Greedy algorithm (not globally optimal)
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("See how a Decision Tree carves the feature space into rectangular decision regions.")
        col1, col2 = st.columns([1, 3])
        with col1:
            max_depth = st.slider("Max Depth", 1, 8, 3, key="dt_depth")
            n_pts = st.slider("Data Points", 100, 400, 200, key="dt_pts")
        with col2:
            np.random.seed(0)
            X, y = make_classification(n_samples=n_pts, n_features=2, n_redundant=0,
                                       n_informative=2, random_state=42, n_clusters_per_class=1)
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
            clf.fit(X, y)

            xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                                  np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig = go.Figure()
            fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z,
                colorscale=[[0,'rgba(26,86,219,0.25)'],[1,'rgba(239,68,68,0.25)']],
                showscale=False, contours=dict(coloring='fill')))
            fig.add_trace(go.Scatter(x=X[y==0,0], y=X[y==0,1], mode='markers',
                marker=dict(color='#1A56DB', size=7, line=dict(color='#fff',width=0.5)), name='Class 0'))
            fig.add_trace(go.Scatter(x=X[y==1,0], y=X[y==1,1], mode='markers',
                marker=dict(color='#EF4444', size=7, line=dict(color='#fff',width=0.5)), name='Class 1'))
            acc = clf.score(X, y)
            fig.update_layout(
                title=dict(text=f"Decision Boundary — Depth {max_depth} | Accuracy: {acc:.2%}", font=dict(color='#E2EAF4')),
                paper_bgcolor='#162032', plot_bgcolor='#0A1220',
                font=dict(color='#E2EAF4'),
                xaxis=dict(gridcolor='#1E3A5F', title='Feature 1'),
                yaxis=dict(gridcolor='#1E3A5F', title='Feature 2'),
                legend=dict(bgcolor='#0A1220'),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"📊 Increase depth → more complex boundary → more overfitting. Current accuracy: **{acc:.2%}**")

    with tab3:
        st.markdown("Configure and train your own Decision Tree on a synthetic dataset.")
        col1, col2, col3 = st.columns(3)
        with col1: depth = st.slider("Tree Depth", 1, 10, 4, key="exp_depth")
        with col2: criterion = st.selectbox("Split Criterion", ["gini", "entropy"])
        with col3: n_data = st.slider("Dataset Size", 50, 500, 200, key="exp_n")

        np.random.seed(1)
        X2, y2 = make_classification(n_samples=n_data, n_features=2, n_redundant=0,
                                      n_informative=2, random_state=5, n_clusters_per_class=1)
        clf2 = DecisionTreeClassifier(max_depth=depth, criterion=criterion)
        clf2.fit(X2, y2)

        xx2, yy2 = np.meshgrid(np.linspace(X2[:,0].min()-0.5, X2[:,0].max()+0.5, 150),
                                 np.linspace(X2[:,1].min()-0.5, X2[:,1].max()+0.5, 150))
        Z2 = clf2.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)

        fig2 = go.Figure()
        fig2.add_trace(go.Contour(x=xx2[0], y=yy2[:,0], z=Z2,
            colorscale=[[0,'rgba(139,92,246,0.25)'],[1,'rgba(16,185,129,0.25)']],
            showscale=False, contours=dict(coloring='fill')))
        fig2.add_trace(go.Scatter(x=X2[y2==0,0], y=X2[y2==0,1], mode='markers',
            marker=dict(color='#8B5CF6', size=7), name='Class 0'))
        fig2.add_trace(go.Scatter(x=X2[y2==1,0], y=X2[y2==1,1], mode='markers',
            marker=dict(color='#10B981', size=7), name='Class 1'))
        fig2.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'), xaxis=dict(gridcolor='#1E3A5F'),
            yaxis=dict(gridcolor='#1E3A5F'), legend=dict(bgcolor='#0A1220'), height=380)
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{clf2.score(X2,y2):.2%}")
        c2.metric("Tree Nodes", clf2.tree_.node_count)
        c3.metric("Tree Depth", clf2.get_depth())
