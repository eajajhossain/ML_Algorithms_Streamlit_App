import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def render():
    st.markdown('<div class="section-title" style="margin-top:0">📐 Linear Regression</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Concept", "🎬 Animated Demo", "🧪 Live Experiment"])

    # ── TAB 1: CONCEPT ────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            <div class="algo-card" style="margin-bottom:1rem;">
                <div class="algo-card-accent" style="background:#10B981;"></div>
                <div class="algo-card-title" style="color:#10B981; margin-left:1rem;">What is Linear Regression?</div>
                <div class="algo-card-desc" style="margin-left:1rem; font-size:0.93rem; color:#E2EAF4; margin-top:0.5rem;">
                    Linear Regression is the foundational <strong>supervised learning</strong> algorithm for predicting 
                    a continuous numeric output. It assumes a linear relationship between the input features 
                    and the target variable, finding the best-fit line that minimizes prediction error.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="formula-box">ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Cost J(θ) = (1/2m) Σ (ŷᵢ − yᵢ)²</div>', unsafe_allow_html=True)

            st.markdown('<div style="font-size:0.85rem; color:#7A9CC0; margin-top:0.8rem;">Where: ŷ = prediction, y = actual, m = number of samples</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title" style="font-size:1rem;">Algorithm Steps</div>', unsafe_allow_html=True)
            steps = [
                ("Initialize", "Start with random weights β₀ (bias) and β₁ (slope)."),
                ("Predict", "Compute ŷ = β₀ + β₁x for all training samples."),
                ("Compute Error", "Calculate Mean Squared Error (MSE) between ŷ and y."),
                ("Gradient Descent", "Compute ∂J/∂β and update: β = β − α·∂J/∂β"),
                ("Repeat", "Iterate until the cost J converges to a minimum."),
            ]
            for i, (title, desc) in enumerate(steps):
                st.markdown(f"""
                <div class="step-box">
                    <div class="step-num">{i+1}</div>
                    <div class="step-text"><strong style="color:#fff;">{title}:</strong> {desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Real-World Use Cases</div>', unsafe_allow_html=True)
        cases = [("🏠","House Price Prediction","Area, rooms → price"),("📈","Stock Forecasting","Historical data → future price"),("🌡️","Weather Prediction","Humidity, pressure → temperature"),("💰","Salary Estimation","Experience, skills → salary")]
        c1,c2,c3,c4 = st.columns(4)
        for col, (icon, title, sub) in zip([c1,c2,c3,c4], cases):
            with col:
                st.markdown(f'<div class="metric-card"><div style="font-size:1.8rem;">{icon}</div><div style="font-size:0.9rem;font-weight:600;color:#fff;margin-top:0.4rem;">{title}</div><div style="font-size:0.78rem;color:#7A9CC0;">{sub}</div></div>', unsafe_allow_html=True)

    # ── TAB 2: ANIMATED DEMO ─────────────────────────────────────────────────
    with tab2:
        st.markdown("Watch how gradient descent moves the regression line towards the optimal fit step by step.")
        np.random.seed(42)
        n = 40
        X_raw = np.linspace(1, 10, n)
        y_raw = 2.5 * X_raw + 1.5 + np.random.randn(n) * 2.5

        step = st.slider("🎬 Gradient Descent Step", 0, 30, 0, key="lr_step")
        alpha = 0.01
        b0, b1 = 0.0, 0.0
        for _ in range(step):
            y_pred = b0 + b1 * X_raw
            db0 = -(2/n) * np.sum(y_raw - y_pred)
            db1 = -(2/n) * np.sum((y_raw - y_pred) * X_raw)
            b0 -= alpha * db0
            b1 -= alpha * db1

        y_line = b0 + b1 * X_raw
        mse = np.mean((y_raw - y_line)**2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_raw, y=y_raw, mode='markers',
            marker=dict(color='#00C2FF', size=8, line=dict(color='#0A1220', width=1)),
            name='Training Data'))
        fig.add_trace(go.Scatter(x=X_raw, y=y_line, mode='lines',
            line=dict(color='#10B981', width=3), name=f'Fitted Line (step {step})'))
        for xi, yi, yp in zip(X_raw, y_raw, y_line):
            fig.add_trace(go.Scatter(x=[xi,xi], y=[yi,yp], mode='lines',
                line=dict(color='#EF4444', width=0.8, dash='dot'),
                showlegend=False))
        fig.update_layout(
            title=dict(text=f"Step {step} | β₀={b0:.3f}  β₁={b1:.3f}  MSE={mse:.3f}", font=dict(color='#E2EAF4')),
            paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'),
            xaxis=dict(gridcolor='#1E3A5F', title='X'),
            yaxis=dict(gridcolor='#1E3A5F', title='y'),
            legend=dict(bgcolor='#0A1220', bordercolor='#1E3A5F'),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"📉 **MSE at step {step}:** `{mse:.4f}` — Move the slider to watch the line converge!")

    # ── TAB 3: LIVE EXPERIMENT ───────────────────────────────────────────────
    with tab3:
        st.markdown("Adjust parameters and generate your own regression experiment.")
        col1, col2, col3 = st.columns(3)
        with col1: n_pts = st.slider("Data Points", 20, 200, 60)
        with col2: noise = st.slider("Noise Level", 0.5, 10.0, 3.0)
        with col3: true_slope = st.slider("True Slope", -5.0, 5.0, 2.0)

        np.random.seed(7)
        X_e = np.linspace(0, 10, n_pts)
        y_e = true_slope * X_e + 3 + np.random.randn(n_pts) * noise

        model = LinearRegression()
        model.fit(X_e.reshape(-1,1), y_e)
        y_fit = model.predict(X_e.reshape(-1,1))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=X_e, y=y_e, mode='markers', marker=dict(color='#00C2FF', size=7), name='Data'))
        fig2.add_trace(go.Scatter(x=X_e, y=y_fit, mode='lines', line=dict(color='#F59E0B', width=3), name='Regression Line'))
        fig2.update_layout(
            paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'),
            xaxis=dict(gridcolor='#1E3A5F'),
            yaxis=dict(gridcolor='#1E3A5F'),
            legend=dict(bgcolor='#0A1220'),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Learned Slope (β₁)", f"{model.coef_[0]:.4f}", f"True: {true_slope}")
        c2.metric("Intercept (β₀)", f"{model.intercept_:.4f}", "True: 3.0")
        r2 = model.score(X_e.reshape(-1,1), y_e)
        c3.metric("R² Score", f"{r2:.4f}", "1.0 = perfect fit")
