import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Animated Presentation on<br>Machine Learning Algorithms</div>
        <div class="hero-subtitle">An interactive guide to the most important ML algorithms — with visualizations, step-by-step animations, and live code demos.</div>
        <span class="hero-badge">Python</span>
        <span class="hero-badge">Streamlit</span>
        <span class="hero-badge">Plotly</span>
        <span class="hero-badge">Scikit-learn</span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Algorithms Covered</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">ML Categories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">∞</div><div class="metric-label">Interactive Charts</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">100%</div><div class="metric-label">Live Code Demos</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Algorithms in this Presentation</div>', unsafe_allow_html=True)

    algorithms = [
        ("📐", "Linear Regression", "#10B981", "Supervised", "Predicts continuous values by fitting a line through data using gradient descent to minimize error."),
        ("🌳", "Decision Trees", "#F59E0B", "Supervised", "Recursively splits data using Gini impurity or Information Gain to build a tree of decisions."),
        ("⭕", "K-Means Clustering", "#8B5CF6", "Unsupervised", "Groups unlabeled data into K clusters by iteratively assigning points to nearest centroids."),
        ("🔷", "Support Vector Machine", "#00C2FF", "Supervised", "Finds the maximum-margin hyperplane separating classes; uses kernel trick for non-linear data."),
        ("🧠", "Neural Networks", "#EF4444", "Deep Learning", "Interconnected layers of neurons learn complex patterns through forward pass and backpropagation."),
        ("🌲", "Random Forest", "#1A56DB", "Ensemble", "Combines many decision trees trained on bootstrap samples and averages their predictions."),
    ]

    col_a, col_b = st.columns(2)
    for i, (icon, name, color, tag, desc) in enumerate(algorithms):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""
            <div class="algo-card">
                <div class="algo-card-accent" style="background:{color};"></div>
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
                    <span style="font-size:1.4rem;">{icon}</span>
                    <span class="algo-card-title">{name}</span>
                    <span style="margin-left:auto; font-size:0.72rem; color:{color}; font-weight:600; background:rgba(255,255,255,0.06); padding:0.15rem 0.6rem; border-radius:10px;">{tag}</span>
                </div>
                <div class="algo-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Radar chart overview
    st.markdown('<div class="section-title">Algorithm Capability Radar</div>', unsafe_allow_html=True)

    categories = ['Accuracy', 'Speed', 'Interpretability', 'Scalability', 'Non-Linear']
    algos_radar = {
        'Linear Reg.':    [6, 9, 10, 9, 2],
        'Decision Tree':  [7, 8, 9,  7, 7],
        'K-Means':        [6, 8, 7,  8, 5],
        'SVM':            [9, 5, 5,  5, 9],
        'Neural Net':     [10,6, 2,  9, 10],
        'Random Forest':  [9, 7, 6,  8, 9],
    }
    colors = ['#10B981','#F59E0B','#8B5CF6','#00C2FF','#EF4444','#1A56DB']

    fig = go.Figure()
    for (name, vals), color in zip(algos_radar.items(), colors):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            line_color=color,
            fillcolor=color,
            opacity=0.18,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='#0A1220',
            radialaxis=dict(visible=True, range=[0,10], color='#7A9CC0', gridcolor='#1E3A5F'),
            angularaxis=dict(color='#E2EAF4', gridcolor='#1E3A5F'),
        ),
        showlegend=True,
        paper_bgcolor='#162032',
        plot_bgcolor='#162032',
        font=dict(color='#E2EAF4', family='DM Sans'),
        legend=dict(bgcolor='#0A1220', bordercolor='#1E3A5F', borderwidth=1),
        margin=dict(t=40, b=40),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="background:#0A1220; border:1px solid #1E3A5F; border-radius:10px; padding:1rem 1.5rem; margin-top:1rem; font-size:0.88rem; color:#7A9CC0;">
        💡 <strong style="color:#00C2FF;">How to use:</strong> Use the sidebar to navigate between algorithms.
        Each page includes an explanation, animated visualization, step-by-step breakdown, live demo with adjustable parameters, and use cases.
    </div>
    """, unsafe_allow_html=True)
