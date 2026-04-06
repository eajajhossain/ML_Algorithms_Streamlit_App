import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles

def render():
    st.markdown('<div class="section-title" style="margin-top:0">🔷 Support Vector Machine (SVM)</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Concept", "🎬 Margin Visualization", "🧪 Live Experiment"])

    with tab1:
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            <div class="algo-card">
                <div class="algo-card-accent" style="background:#00C2FF;"></div>
                <div class="algo-card-title" style="color:#00C2FF; margin-left:1rem;">What is SVM?</div>
                <div class="algo-card-desc" style="margin-left:1rem; font-size:0.93rem; color:#E2EAF4; margin-top:0.5rem;">
                    Support Vector Machines find the <strong>optimal hyperplane</strong> that maximizes the margin 
                    between two classes. The data points closest to the boundary are called 
                    <strong>support vectors</strong> — they define and support the decision boundary.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Decision: f(x) = w·x + b</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Maximize Margin: 2 / ‖w‖</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Kernel: K(x,z) = φ(x)·φ(z)</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title" style="font-size:1rem;">Key Concepts</div>', unsafe_allow_html=True)
            concepts = [
                ("Hyperplane", "A decision boundary that separates the two classes."),
                ("Margin", "The gap between the hyperplane and the nearest support vectors."),
                ("Support Vectors", "The critical data points that define the margin boundaries."),
                ("C Parameter", "Controls trade-off: small C = wide margin, large C = fewer misclassifications."),
                ("Kernel Trick", "Maps data to higher dimensions to make it linearly separable."),
            ]
            for i, (title, desc) in enumerate(concepts):
                st.markdown(f"""
                <div class="step-box">
                    <div class="step-num">{i+1}</div>
                    <div class="step-text"><strong style="color:#fff;">{title}:</strong> {desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("Visualize the decision boundary and support vectors for different C values.")
        col_ctrl, col_plot = st.columns([1, 2.5])
        with col_ctrl:
            C_val = st.select_slider("C (Regularization)", [0.01, 0.1, 1, 10, 100], value=1)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], key="svm_k")

        np.random.seed(5)
        X, y = make_classification(n_samples=120, n_features=2, n_redundant=0,
                                    n_informative=2, random_state=5, n_clusters_per_class=1)
        svm = SVC(C=C_val, kernel=kernel, gamma='scale')
        svm.fit(X, y)

        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                              np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        decision = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        with col_plot:
            fig = go.Figure()
            fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z,
                colorscale=[[0,'rgba(26,86,219,0.2)'],[1,'rgba(239,68,68,0.2)']],
                showscale=False, contours=dict(coloring='fill')))
            fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=decision,
                colorscale=[[0,'rgba(0,194,255,0)'],[0.5,'rgba(0,194,255,0.6)'],[1,'rgba(0,194,255,0)']],
                showscale=False, contours=dict(start=-1, end=1, size=2, coloring='lines'),
                line=dict(dash='dash', color='#00C2FF', width=1.5)))
            fig.add_trace(go.Scatter(x=X[y==0,0], y=X[y==0,1], mode='markers',
                marker=dict(color='#1A56DB', size=7), name='Class 0'))
            fig.add_trace(go.Scatter(x=X[y==1,0], y=X[y==1,1], mode='markers',
                marker=dict(color='#EF4444', size=7), name='Class 1'))
            sv = svm.support_vectors_
            fig.add_trace(go.Scatter(x=sv[:,0], y=sv[:,1], mode='markers',
                marker=dict(color='rgba(0,0,0,0)', size=14,
                            line=dict(color='#F59E0B', width=2.5)), name='Support Vectors'))
            acc = svm.score(X, y)
            fig.update_layout(
                title=dict(text=f"Kernel={kernel} | C={C_val} | Accuracy={acc:.2%}", font=dict(color='#E2EAF4')),
                paper_bgcolor='#162032', plot_bgcolor='#0A1220',
                font=dict(color='#E2EAF4'),
                xaxis=dict(gridcolor='#1E3A5F', title='Feature 1'),
                yaxis=dict(gridcolor='#1E3A5F', title='Feature 2'),
                legend=dict(bgcolor='#0A1220'),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
            n_sv = len(svm.support_vectors_)
            st.info(f"🔶 **Support Vectors:** {n_sv} | **Accuracy:** {acc:.2%} | Dashed lines = margin boundaries")

    with tab3:
        st.markdown("Test SVM with the **Kernel Trick** on non-linearly separable (circular) data.")
        col1, col2 = st.columns(2)
        with col1:
            kernel2 = st.selectbox("Kernel", ["linear", "rbf", "poly"], key="svm_k2")
        with col2:
            C2 = st.select_slider("C", [0.1, 1, 10, 100], value=1, key="svm_c2")

        np.random.seed(3)
        X2, y2 = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=3)
        svm2 = SVC(C=C2, kernel=kernel2, gamma='scale')
        svm2.fit(X2, y2)

        xx2, yy2 = np.meshgrid(np.linspace(X2[:,0].min()-0.3, X2[:,0].max()+0.3, 200),
                                 np.linspace(X2[:,1].min()-0.3, X2[:,1].max()+0.3, 200))
        Z2 = svm2.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)

        fig2 = go.Figure()
        fig2.add_trace(go.Contour(x=xx2[0], y=yy2[:,0], z=Z2,
            colorscale=[[0,'rgba(139,92,246,0.2)'],[1,'rgba(16,185,129,0.2)']],
            showscale=False, contours=dict(coloring='fill')))
        fig2.add_trace(go.Scatter(x=X2[y2==0,0], y=X2[y2==0,1], mode='markers',
            marker=dict(color='#8B5CF6', size=8), name='Class 0'))
        fig2.add_trace(go.Scatter(x=X2[y2==1,0], y=X2[y2==1,1], mode='markers',
            marker=dict(color='#10B981', size=8), name='Class 1'))
        acc2 = svm2.score(X2, y2)
        fig2.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'), xaxis=dict(gridcolor='#1E3A5F'),
            yaxis=dict(gridcolor='#1E3A5F'), legend=dict(bgcolor='#0A1220'), height=380)
        st.plotly_chart(fig2, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc2:.2%}")
        c2.metric("Support Vectors", len(svm2.support_vectors_))
        if kernel2 == "linear" and acc2 < 0.75:
            st.warning("⚠️ Linear kernel struggles with circular data. Try **rbf** kernel!")
        elif acc2 > 0.9:
            st.success(f"✅ {kernel2.upper()} kernel handles non-linear boundaries well!")
