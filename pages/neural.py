import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

def render():
    st.markdown('<div class="section-title" style="margin-top:0">🧠 Neural Networks</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Concept", "🎬 Network Architecture", "🧪 Live Training"])

    with tab1:
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            <div class="algo-card">
                <div class="algo-card-accent" style="background:#EF4444;"></div>
                <div class="algo-card-title" style="color:#EF4444; margin-left:1rem;">What is a Neural Network?</div>
                <div class="algo-card-desc" style="margin-left:1rem; font-size:0.93rem; color:#E2EAF4; margin-top:0.5rem;">
                    Inspired by the human brain, Neural Networks are composed of 
                    <strong>layers of interconnected neurons</strong>. Each neuron computes a weighted sum 
                    of its inputs, applies an activation function, and passes the result to the next layer.
                    They learn by adjusting weights via <strong>backpropagation</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Neuron: z = Σ (wᵢ · xᵢ) + b</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Activation: a = σ(z)  e.g. ReLU, Sigmoid</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Loss: L = −Σ yᵢ·log(ŷᵢ)</div>', unsafe_allow_html=True)
            st.markdown('<div class="formula-box">Weight Update: w = w − α·∂L/∂w</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-title" style="font-size:1rem;">Training Pipeline</div>', unsafe_allow_html=True)
            steps = [
                ("Forward Pass", "Input flows through each layer computing activations."),
                ("Compute Loss", "Compare final output to ground truth using a loss function."),
                ("Backpropagation", "Compute gradient of loss w.r.t. every weight using chain rule."),
                ("Update Weights", "Gradient descent: w = w − α·∂L/∂w"),
                ("Epoch", "One full pass over the training data. Repeat for many epochs."),
            ]
            for i, (title, desc) in enumerate(steps):
                st.markdown(f"""
                <div class="step-box">
                    <div class="step-num">{i+1}</div>
                    <div class="step-text"><strong style="color:#fff;">{title}:</strong> {desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("Visualize a neural network's architecture with configurable layers.")

        col1, col2 = st.columns([1, 3])
        with col1:
            n_hidden = st.slider("Hidden Layers", 1, 3, 2)
            neurons_per = st.slider("Neurons/Layer", 2, 8, 4)
        with col2:
            layer_sizes = [3] + [neurons_per]*n_hidden + [2]
            n_layers = len(layer_sizes)

            fig = go.Figure()
            layer_labels = ['Input'] + [f'Hidden {i+1}' for i in range(n_hidden)] + ['Output']
            layer_colors = ['#10B981'] + ['#1A56DB']*n_hidden + ['#EF4444']
            node_positions = []

            for li, (n_neurons, color) in enumerate(zip(layer_sizes, layer_colors)):
                x = li / (n_layers - 1) * 9 + 0.5
                positions = []
                for ni in range(n_neurons):
                    y = (ni - (n_neurons-1)/2) * 1.2 + 3
                    positions.append((x, y))
                node_positions.append(positions)

            # Connections
            for li in range(n_layers - 1):
                for (x1, y1) in node_positions[li]:
                    for (x2, y2) in node_positions[li+1]:
                        weight = np.random.uniform(0.1, 1)
                        fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode='lines',
                            line=dict(color=f'rgba(26,86,219,{weight*0.4:.2f})', width=weight*1.5),
                            showlegend=False))

            # Nodes
            for li, (positions, color, label) in enumerate(zip(node_positions, layer_colors, layer_labels)):
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers',
                    marker=dict(color=color, size=22, line=dict(color='white', width=2)),
                    name=label))
                fig.add_annotation(x=xs[0], y=min(ys)-0.8, text=label,
                    showarrow=False, font=dict(color='#7A9CC0', size=10))

            fig.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
                font=dict(color='#E2EAF4'),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                showlegend=True, legend=dict(bgcolor='#0A1220'),
                height=400, margin=dict(t=30, b=10),
                title=dict(text=f"Architecture: {' → '.join(map(str, layer_sizes))}", font=dict(color='#E2EAF4'))
            )
            st.plotly_chart(fig, use_container_width=True)
            total_params = sum(layer_sizes[i]*layer_sizes[i+1] + layer_sizes[i+1] for i in range(n_layers-1))
            st.info(f"🔢 Total trainable parameters: **{total_params}** (weights + biases)")

    with tab3:
        st.markdown("Train an MLP on non-linear (moon-shaped) data and watch the decision boundary evolve.")
        col1, col2, col3 = st.columns(3)
        with col1:
            hidden_size = st.slider("Hidden Layer Size", 4, 32, 10)
        with col2:
            max_iter = st.slider("Training Epochs", 10, 500, 100)
        with col3:
            lr = st.select_slider("Learning Rate", [0.001, 0.01, 0.05, 0.1, 0.5], value=0.01)

        np.random.seed(0)
        X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=0)
        mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=max_iter,
                            learning_rate_init=lr, random_state=0, activation='relu')
        mlp.fit(X_m, y_m)

        xx, yy = np.meshgrid(np.linspace(X_m[:,0].min()-0.3, X_m[:,0].max()+0.3, 200),
                              np.linspace(X_m[:,1].min()-0.3, X_m[:,1].max()+0.3, 200))
        Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig2 = go.Figure()
        fig2.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z,
            colorscale=[[0,'rgba(239,68,68,0.25)'],[1,'rgba(0,194,255,0.25)']],
            showscale=False, contours=dict(coloring='fill')))
        fig2.add_trace(go.Scatter(x=X_m[y_m==0,0], y=X_m[y_m==0,1], mode='markers',
            marker=dict(color='#EF4444', size=7), name='Class 0'))
        fig2.add_trace(go.Scatter(x=X_m[y_m==1,0], y=X_m[y_m==1,1], mode='markers',
            marker=dict(color='#00C2FF', size=7), name='Class 1'))
        fig2.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
            font=dict(color='#E2EAF4'), xaxis=dict(gridcolor='#1E3A5F'),
            yaxis=dict(gridcolor='#1E3A5F'), legend=dict(bgcolor='#0A1220'), height=370)
        st.plotly_chart(fig2, use_container_width=True)

        acc = mlp.score(X_m, y_m)
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Accuracy", f"{acc:.2%}")
        c2.metric("Epochs Run", mlp.n_iter_)
        c3.metric("Loss (final)", f"{mlp.loss_:.4f}")

        # Loss curve if available
        if hasattr(mlp, 'loss_curve_') and len(mlp.loss_curve_) > 1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(y=mlp.loss_curve_, mode='lines',
                line=dict(color='#EF4444', width=2.5), name='Training Loss'))
            fig3.update_layout(paper_bgcolor='#162032', plot_bgcolor='#0A1220',
                font=dict(color='#E2EAF4'),
                xaxis=dict(title='Epoch', gridcolor='#1E3A5F'),
                yaxis=dict(title='Loss', gridcolor='#1E3A5F'),
                height=250, margin=dict(t=20, b=20),
                title=dict(text="Training Loss Curve", font=dict(color='#E2EAF4')))
            st.plotly_chart(fig3, use_container_width=True)
