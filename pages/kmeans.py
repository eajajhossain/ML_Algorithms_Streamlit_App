import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

COLORS = ['#00C2FF','#10B981','#F59E0B','#EF4444','#8B5CF6']

def render():
    st.markdown('<div class="section-title" style="margin-top:0">⭕ K-Means Clustering</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Concept", "🎬 Animated Steps", "🧪 Live Experiment"])

    with tab1:
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            <div class="algo-card">
                <div class="algo-card-accent" style="background:#8B5CF6;"></div>
                <div class="algo-card-title" style="color:#8B5CF6; margin-left:1rem;">What is K-Means?</div>
                <div class="algo-card-desc" style="margin-left:1rem; font-size:0.93rem; color:#E2EAF4; margin-top:0.5rem;">
                    K-Means is an <strong>unsupervised</strong> clustering algorithm that partitions n observations 
                    into K clusters. It minimizes the within-cluster sum of squared distances (inertia) 
                    between data points and their assigned centroid.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Objective: min Σ Σ ‖xᵢ − μₖ‖²</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">New centroid: μₖ = (1/|Sₖ|) Σ xᵢ</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#0A1220; border:1px solid #8B5CF6; border-radius:8px; padding:1rem; margin-top:1rem; font-size:0.85rem; color:#E2EAF4;">
                <strong style="color:#8B5CF6;">Elbow Method</strong> — To choose optimal K, plot inertia vs K values. 
                The "elbow" point where inertia stops decreasing sharply is the best K.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-title" style="font-size:1rem;">Algorithm Steps</div>', unsafe_allow_html=True)
            steps = [
                ("Choose K", "Decide the number of clusters K upfront."),
                ("Initialize", "Randomly place K centroids in the feature space."),
                ("Assign", "Assign each point to the nearest centroid (by Euclidean distance)."),
                ("Update", "Recompute each centroid as the mean of its assigned points."),
                ("Repeat", "Repeat Assign & Update until centroids stop moving."),
            ]
            for i, (title, desc) in enumerate(steps):
                st.markdown(f"""
                <div class="step-box">
                    <div class="step-num">{i+1}</div>
                    <div class="step-text"><strong style="color:#fff;">{title}:</strong> {desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("Watch centroids evolve iteration by iteration until convergence.")
        col_ctrl, col_plot = st.columns([1, 2.5])
        with col_ctrl:
            k_val = st.slider("K (clusters)", 2, 5, 3, key="km_k")
            iteration = st.slider("Iteration", 1, 10, 1, key="km_iter")
            np.random.seed(42)

        np.random.seed(42)
        X_blobs, _ = make_blobs(n_samples=150, centers=k_val, cluster_std=1.2, random_state=42)

        # Manual K-Means for animation
        centroids = X_blobs[np.random.choice(len(X_blobs), k_val, replace=False)]
        for _ in range(iteration):
            dists = np.linalg.norm(X_blobs[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([X_blobs[labels==i].mean(axis=0) if (labels==i).sum() > 0 else centroids[i] for i in range(k_val)])
            centroids = new_centroids

        with col_plot:
            fig = go.Figure()
            for i in range(k_val):
                pts = X_blobs[labels==i]
                fig.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='markers',
                    marker=dict(color=COLORS[i], size=7, opacity=0.75),
                    name=f'Cluster {i+1}'))
                fig.add_trace(go.Scatter(x=[centroids[i,0]], y=[centroids[i,1]], mode='markers',
                    marker=dict(color=COLORS[i], size=18, symbol='star',
                                line=dict(color='white', width=2)),
                    name=f'Centroid {i+1}'))
            fig.update_layout(
                title=dict(text=f"K={k_val} | Iteration {iteration}", font=dict(color='#E2EAF4')),
                paper_bgcolor='#162032', plot_bgcolor='#0A1220',
                font=dict(color='#E2EAF4'),
                xaxis=dict(gridcolor='#1E3A5F'),
                yaxis=dict(gridcolor='#1E3A5F'),
                legend=dict(bgcolor='#0A1220'),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("Explore K-Means with full sklearn and the Elbow Method.")
        col1, col2, col3 = st.columns(3)
        with col1: k = st.slider("K (clusters)", 2, 8, 3, key="ex_k")
        with col2: n_centers = st.slider("True Centers", 2, 6, 3, key="ex_c")
        with col3: spread = st.slider("Cluster Spread", 0.5, 3.0, 1.2, key="ex_s")

        np.random.seed(10)
        X_e, y_true = make_blobs(n_samples=250, centers=n_centers, cluster_std=spread, random_state=10)
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        km.fit(X_e)

        fig2 = go.Figure()
        for i in range(k):
            pts = X_e[km.labels_==i]
            fig2.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='markers',
                marker=dict(color=COLORS[i%5], size=7), name=f'Cluster {i+1}'))
            cx, cy = km.cluster_centers_[i]
            fig2.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers',
                marker=dict(color=COLORS[i%5], size=16, symbol='star', line=dict(color='white',width=2)),
                name=f'C{i+1} center', showlegend=False))
        fig2.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'), xaxis=dict(gridcolor='#1E3A5F'),
            yaxis=dict(gridcolor='#1E3A5F'), legend=dict(bgcolor='#0A1220'), height=350)
        st.plotly_chart(fig2, use_container_width=True)

        # Elbow
        st.markdown('<div class="section-title" style="font-size:1rem;">📐 Elbow Method</div>', unsafe_allow_html=True)
        inertias = [KMeans(n_clusters=i, random_state=0, n_init=10).fit(X_e).inertia_ for i in range(1, 10)]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=list(range(1,10)), y=inertias, mode='lines+markers',
            line=dict(color='#8B5CF6', width=2.5),
            marker=dict(color='#8B5CF6', size=8, line=dict(color='white',width=1))))
        fig3.add_vline(x=n_centers, line_dash='dash', line_color='#F59E0B', annotation_text=f"True K={n_centers}")
        fig3.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'), xaxis=dict(title='K', gridcolor='#1E3A5F'),
            yaxis=dict(title='Inertia', gridcolor='#1E3A5F'), height=280, margin=dict(t=30))
        st.plotly_chart(fig3, use_container_width=True)
        c1,c2 = st.columns(2)
        c1.metric("Inertia (lower=better)", f"{km.inertia_:.1f}")
        c2.metric("Iterations to converge", km.n_iter_)
