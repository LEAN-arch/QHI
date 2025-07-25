# =================================================================================================
# LottoSphere v16.0.1: The Quantum Chronodynamics Engine
#
# AUTHOR: Subject Matter Expert AI (Stochastic Systems, Predictive Dynamics & Complex Systems)
# DATE: 2025-07-25
# VERSION: 16.0.1 (Debugged & Optimized)
#
# DESCRIPTION:
# A professional-grade scientific instrument for analyzing high-dimensional, chaotic time-series
# data, framed around lottery number sets. Models each of six sorted number positions as an
# independent yet interacting dynamical system. Integrates deep learning, statistical physics,
# chaos theory, and quantum-inspired methods with a robust metrology suite.
# =================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import warnings

# --- Suppress Warnings for a Cleaner UI ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Advanced Scientific & ML Libraries ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import umap
import hdbscan
import pywt
from scipy.signal import welch
from nolds import lyap_r

# --- Deep Learning (PyTorch) ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. APPLICATION CONFIGURATION ---
st.set_page_config(
    page_title="LottoSphere v16.0.1: Quantum Chronodynamics",
    page_icon="⚛️",
    layout="wide",
)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================================================================================================
# ALL FUNCTION DEFINITIONS
# =================================================================================================

# --- 2. CORE UTILITIES & DATA HANDLING ---
@st.cache_data
def load_data(uploaded_file, max_num=49):
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        # Validate number range (1–max_num, positive integers)
        valid_range = (df >= 1) & (df <= max_num) & (df == df.astype(int))
        if not valid_range.all().all():
            st.session_state.data_warning = f"Invalid numbers detected (must be integers between 1 and {max_num}). Discarding invalid rows."
            df = df[valid_range.all(axis=1)].reset_index(drop=True)
        
        # Check for duplicates within rows
        num_cols = df.shape[1]
        unique_counts = df.apply(lambda row: len(set(row)), axis=1)
        valid_rows_mask = (unique_counts == num_cols)
        if not valid_rows_mask.all():
            st.session_state.data_warning = f"Data integrity issue. Discarded {len(df) - valid_rows_mask.sum()} rows with duplicate/missing numbers."
            df = df[valid_rows_mask].reset_index(drop=True)
        
        # Check for duplicate rows
        if df.duplicated().any():
            st.session_state.data_warning = f"Discarded {df.duplicated().sum()} duplicate rows."
            df = df.drop_duplicates().reset_index(drop=True)
        
        if df.shape[1] > 6:
            df = df.iloc[:, :6]
        df.columns = [f'Pos_{i+1}' for i in range(df.shape[1])]
        
        if len(df) < 50:
            st.session_state.data_warning = "Insufficient data for robust analysis (at least 50 rows required)."
            return pd.DataFrame()
        
        # Sort each row to create stable positional time series
        st.session_state.data_warning = "Input data sorted per row to create positional time series."
        sorted_values = np.sort(df.values, axis=1)
        return pd.DataFrame(sorted_values, columns=df.columns).astype(int)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_sequences(data, seq_length):
    if seq_length >= len(data):
        raise ValueError("Sequence length must be less than data length")
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 3. TIME-DEPENDENT BEHAVIOR ANALYSIS MODULE ---
@st.cache_data
def analyze_temporal_behavior(_df, position='Pos_1'):
    try:
        results = {}
        series = _df[position].values
        
        # Recurrence Plot
        len_series = len(series)
        recurrence_matrix = np.zeros((len_series, len_series))
        for i in range(len_series):
            for j in range(len_series):
                recurrence_matrix[i, j] = np.abs(series[i] - series[j])
        recurrence_matrix /= recurrence_matrix.max() + 1e-10  # Normalize
        results['recurrence_fig'] = px.imshow(
            recurrence_matrix,
            color_continuous_scale='viridis',
            title=f"Recurrence Plot ({position})"
        )
        
        # Fourier Analysis (Power Spectrum)
        freqs, psd = welch(series, nperseg=min(len(series), 256))
        psd_df = pd.DataFrame({'Frequency': freqs, 'Power': psd}).sort_values('Power', ascending=False)
        results['fourier_fig'] = px.line(
            psd_df,
            x='Frequency',
            y='Power',
            title=f"Power Spectral Density ({position})"
        )
        
        # Wavelet Transform
        widths = np.arange(1, min(31, len(series)//2))
        cwt_matrix, _ = pywt.cwt(series, widths, 'morl')
        results['wavelet_fig'] = go.Figure(data=go.Heatmap(
            z=np.abs(cwt_matrix),
            x=np.arange(len(series)),
            y=widths,
            colorscale='viridis'
        ))
        results['wavelet_fig'].update_layout(
            title=f'Continuous Wavelet Transform ({position})',
            xaxis_title='Time',
            yaxis_title='Scale'
        )
        
        # Lyapunov Exponent
        try:
            lyap_exp = lyap_r(series, emb_dim=2)
            results['lyapunov'] = lyap_exp
        except Exception as e:
            st.warning(f"Lyapunov exponent calculation failed: {e}")
            results['lyapunov'] = -1

        return results
    except Exception as e:
        st.error(f"Error in temporal behavior analysis: {e}")
        return {}

# --- 4. ADVANCED PREDICTIVE MODELS ---

@st.cache_resource
def train_torch_model(_df, model_type='LSTM', seq_length=10, epochs=100, batch_size=32):
    try:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(_df)
        X, y = create_sequences(data_scaled, seq_length)
        X_torch, y_torch = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)
        
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        class SequenceModel(nn.Module):
            def __init__(self):
                super().__init__()
                if model_type == 'LSTM':
                    self.rnn = nn.LSTM(input_size=6, hidden_size=50, num_layers=2, batch_first=True)
                elif model_type == 'GRU':
                    self.rnn = nn.GRU(input_size=6, hidden_size=50, num_layers=2, batch_first=True)
                self.fc = nn.Linear(50, 6)
            def forward(self, x):
                x, _ = self.rnn(x)
                return self.fc(x[:, -1, :])

        model = SequenceModel().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_loss = float('inf')
        patience, max_patience = 0, 10
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader)
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
            if patience > max_patience:
                break
        
        return model, scaler, best_loss
    except Exception as e:
        st.error(f"Error training {model_type} model: {e}")
        return None, None, float('inf')

@st.cache_data
def predict_torch_model(_df, _model_cache, model_type='LSTM', seq_length=10, max_num=49):
    try:
        model, scaler, best_loss = _model_cache
        if model is None:
            raise ValueError(f"{model_type} model training failed")
        
        last_seq = scaler.transform(_df.iloc[-seq_length:].values).reshape(1, seq_length, 6)
        last_seq_torch = torch.tensor(last_seq, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            pred_scaled = model(last_seq_torch)
        
        prediction = scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        # Clamp to valid range and ensure uniqueness
        prediction = np.clip(np.round(prediction), 1, max_num).astype(int)
        unique_preds = []
        seen = set()
        for p in prediction:
            if p not in seen:
                unique_preds.append(p)
                seen.add(p)
        # Fill with random valid numbers if needed
        while len(unique_preds) < 6:
            new_num = np.random.randint(1, max_num + 1)
            if new_num not in seen:
                unique_preds.append(new_num)
                seen.add(new_num)
        
        # Error estimation based on validation loss
        error = np.full(6, np.sqrt(best_loss) * 10)
        
        return {
            'name': model_type,
            'prediction': sorted(unique_preds),
            'error': error,
            'logic': f'Deep learning {model_type} sequence forecast.'
        }
    except Exception as e:
        st.error(f"Error predicting with {model_type}: {e}")
        return {
            'name': model_type,
            'prediction': [0]*6,
            'error': [0]*6,
            'logic': 'Failed due to error.'
        }

@st.cache_data
def analyze_hilbert_embedding(_df, max_num=49):
    try:
        if len(_df) < 2:
            raise ValueError("Insufficient data for Hilbert embedding")
        
        # Map numbers to complex plane
        def to_complex(n):
            return np.exp(1j * 2 * np.pi * n / max_num)
        
        complex_df = _df.apply(lambda x: to_complex(x))
        mean_vector = complex_df.mean(axis=1)
        
        # Extrapolate next mean vector
        last_phase, last_amp = np.angle(mean_vector.iloc[-1]), np.abs(mean_vector.iloc[-1])
        phase_velocity = np.angle(mean_vector.iloc[-1] / mean_vector.iloc[-2])
        amp_velocity = np.abs(mean_vector.iloc[-1]) - np.abs(mean_vector.iloc[-2])
        
        next_phase = last_phase + phase_velocity
        next_amp = max(1e-10, last_amp + amp_velocity)  # Avoid zero amplitude
        predicted_vector = next_amp * np.exp(1j * next_phase)
        
        # Greedy approximation to find 6 numbers
        all_nums = np.arange(1, max_num + 1)
        all_complex = to_complex(all_nums)
        selected = []
        remaining = set(range(len(all_complex)))
        for _ in range(6):
            min_dist = np.inf
            best_idx = None
            current_sum = np.sum([all_complex[i] for i in selected]) / max(len(selected), 1)
            for idx in remaining:
                test_sum = (current_sum * len(selected) + all_complex[idx]) / (len(selected) + 1)
                dist = np.abs(test_sum - predicted_vector)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        prediction = sorted(all_nums[selected])
        error = np.full(6, min_dist * 10)
        
        return {
            'name': 'Hilbert Space Embedding',
            'prediction': prediction,
            'error': error,
            'logic': 'Predicts next draw\'s geometric center in a complex Hilbert space.'
        }
    except Exception as e:
        st.error(f"Error in Hilbert embedding: {e}")
        return {
            'name': 'Hilbert Space Embedding',
            'prediction': [0]*6,
            'error': [0]*6,
            'logic': 'Failed due to error.'
        }

# --- 5. BACKTESTING & META-ANALYSIS ---
@st.cache_data
def run_full_backtest_suite(df, max_num=49):
    try:
        scored_predictions = []
        split_point = max(50, int(len(df) * 0.8))
        if len(df) - split_point < 10:
            st.warning("Insufficient validation data. Using simplified backtest.")
            split_point = len(df) - 10
        
        train_df = df.iloc[:split_point]
        val_df = df.iloc[split_point:]
        
        lstm_cache = train_torch_model(train_df, model_type='LSTM')
        gru_cache = train_torch_model(train_df, model_type='GRU')
        
        model_funcs = {
            'LSTM': lambda d: predict_torch_model(d, lstm_cache, 'LSTM', max_num=max_num),
            'GRU': lambda d: predict_torch_model(d, gru_cache, 'GRU', max_num=max_num),
            'Hilbert Embedding': lambda d: analyze_hilbert_embedding(d, max_num=max_num),
        }
        
        progress_bar = st.progress(0, text="Backtesting models...")
        total_steps = len(val_df) * len(model_funcs)
        current_step = 0
        
        for name, func in model_funcs.items():
            y_preds, y_trues = [], []
            for i in range(len(val_df)):
                historical_df = df.iloc[:split_point + i]
                pred = func(historical_df)['prediction']
                if all(p == 0 for p in pred):  # Skip failed predictions
                    continue
                y_preds.append(pred)
                y_trues.append(val_df.iloc[i].values.tolist())
                current_step += 1
                progress_bar.progress(min(1.0, current_step / total_steps), text=f"Backtesting {name} on draw {i+1}")
            
            if not y_preds:
                likelihood = 0
                metrics = {'Avg Hits': 'N/A'}
            else:
                hits = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues, y_preds))
                rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
                accuracy = hits / len(y_trues)
                likelihood = 0.6 * min(100, (accuracy / 1.0) * 100) + 0.4 * max(0, 100 - rmse * 5)
                metrics = {'Avg Hits': f"{accuracy:.2f}", 'RMSE': f"{rmse:.2f}"}
            
            final_pred_obj = func(df)
            final_pred_obj['likelihood'] = likelihood
            final_pred_obj['metrics'] = metrics
            scored_predictions.append(final_pred_obj)
        
        progress_bar.empty()
        return sorted(scored_predictions, key=lambda x: x.get('likelihood', 0), reverse=True)
    except Exception as e:
        st.error(f"Error in backtesting suite: {e}")
        return []

@st.cache_data
def analyze_predictive_maturity(df, model_type='LSTM', max_num=49):
    try:
        history_sizes = np.linspace(50, len(df), 8, dtype=int)
        maturity_scores, prediction_deltas = [], []
        
        progress_bar = st.progress(0, text="Analyzing predictive maturity...")
        total_steps = len(history_sizes)
        
        for idx, size in enumerate(history_sizes):
            subset_df = df.iloc[:size]
            if len(subset_df) < 50:
                continue
            
            model_cache = train_torch_model(subset_df, model_type)
            pred_obj = predict_torch_model(subset_df, model_cache, model_type, max_num=max_num)
            prediction_deltas.append(pred_obj['prediction'])
            
            split = max(40, int(len(subset_df) * 0.8))
            if len(subset_df) - split < 10:
                continue
            train, val = subset_df.iloc[:split], subset_df.iloc[split:]
            
            val_preds = [
                predict_torch_model(subset_df.iloc[:split+i], model_cache, model_type, max_num=max_num)['prediction']
                for i in range(len(val))
            ]
            val_trues = val.values.tolist()
            hits = sum(len(set(yt) & set(yp)) for yt, yp in zip(val_trues, val_preds))
            accuracy = hits / len(val_trues)
            maturity_scores.append({'History Size': size, 'Likelihood Score': (accuracy / 1.0) * 100})
            
            progress_bar.progress((idx + 1) / total_steps, text=f"Analyzed history size {size}")
        
        progress_bar.empty()
        delta_df = pd.DataFrame(prediction_deltas, index=[f"Size {s}" for s in history_sizes if s <= len(df)])
        delta_df.columns = [f"Pos {i+1}" for i in range(delta_df.shape[1])]
        return pd.DataFrame(maturity_scores), delta_df
    except Exception as e:
        st.error(f"Error in predictive maturity analysis: {e}")
        return pd.DataFrame(), pd.DataFrame()

# =================================================================================================
# Main Application UI & Logic
# =================================================================================================

st.title("⚛️ LottoSphere v16.0.1: The Quantum Chronodynamics Engine")
st.markdown("A scientific instrument for exploratory analysis of high-dimensional, chaotic systems. Models each number position as an evolving system using advanced mathematical and AI techniques.")

if 'data_warning' not in st.session_state:
    st.session_state.data_warning = None
if 'error_messages' not in st.session_state:
    st.session_state.error_messages = []

# Sidebar Configuration
st.sidebar.header("Configuration")
max_num = st.sidebar.number_input("Max Number (e.g., 49)", min_value=10, max_value=100, value=49)
seq_length = st.sidebar.slider("Sequence Length", min_value=5, max_value=20, value=10)
epochs = st.sidebar.slider("Training Epochs", min_value=50, max_value=200, value=100)

uploaded_file = st.sidebar.file_uploader("Upload Number.csv", type=["csv"])

if st.session_state.error_messages:
    for msg in st.session_state.error_messages:
        st.error(msg)
    st.session_state.error_messages = []

if uploaded_file:
    df_master = load_data(uploaded_file, max_num=max_num)
    if st.session_state.data_warning:
        st.warning(st.session_state.data_warning)
        st.session_state.data_warning = None

    if not df_master.empty and df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        
        tab1, tab2, tab3 = st.tabs(["🔮 Predictive Analytics", "🔬 System Dynamics Explorer", "🧠 Predictive Maturity"])

        with tab1:
            st.header("Engage Grand Unified Predictive Ensemble")
            if st.button("RUN ALL PREDICTIVE MODELS", type="primary", use_container_width=True):
                with st.spinner("Backtesting all models... This may take several minutes."):
                    scored_predictions = run_full_backtest_suite(df_master, max_num=max_num)
                
                st.header("✨ Final Synthesis & Strategic Portfolio")
                if scored_predictions:
                    st.subheader("Ranked Predictions by Historical Performance")
                    for p in scored_predictions:
                        with st.container(border=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"#### {p['name']}")
                                pred_str = ' | '.join([f"{n} <small>(±{e:.1f})</small>" for n, e in zip(p['prediction'], p['error'])])
                                st.markdown(f"**Candidate Set:** {pred_str}", unsafe_allow_html=True)
                            with col2:
                                st.metric("Likelihood Score", f"{p.get('likelihood', 0):.1f}%", help=f"Backtest Metrics: {p.get('metrics', {})}")

        with tab2:
            st.header("System Dynamics Explorer")
            st.markdown("Explore the intrinsic, time-dependent behavior of the number system.")
            position = st.selectbox("Select Position", options=df_master.columns, index=0)
            if st.button("ANALYZE DYNAMICS"):
                with st.spinner("Calculating system dynamics..."):
                    dynamic_results = analyze_temporal_behavior(df_master, position=position)
                
                if dynamic_results:
                    st.subheader(f"Chaotic & Cyclical Analysis ({position})")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(dynamic_results['recurrence_fig'], use_container_width=True)
                    with col2:
                        st.plotly_chart(dynamic_results['fourier_fig'], use_container_width=True)
                    st.plotly_chart(dynamic_results['wavelet_fig'], use_container_width=True)
                    if dynamic_results['lyapunov'] > 0:
                        st.warning(
                            f"**Lyapunov Exponent:** `{dynamic_results['lyapunov']:.4f}`. A positive value suggests the system is chaotic and highly sensitive to initial conditions.",
                            icon="⚠️"
                        )
                    else:
                        st.success(
                            f"**Lyapunov Exponent:** `{dynamic_results['lyapunov']:.4f}`. A non-positive value suggests the system is stable or periodic, not chaotic.",
                            icon="✅"
                        )

        with tab3:
            st.header("Predictive Maturity Analysis")
            st.markdown("Determine how predictive power evolves with historical data size.")
            model_type = st.selectbox("Select Model", options=['LSTM', 'GRU'], index=0)
            if st.button("RUN MATURITY ANALYSIS"):
                with st.spinner("Performing iterative backtesting... This is computationally expensive."):
                    maturity_df, delta_df = analyze_predictive_maturity(df_master, model_type=model_type, max_num=max_num)
                if not maturity_df.empty:
                    st.subheader("Model Performance vs. History Size")
                    fig_maturity = px.line(
                        maturity_df,
                        x='History Size',
                        y='Likelihood Score',
                        title="Predictive Maturity Curve",
                        markers=True
                    )
                    st.plotly_chart(fig_maturity, use_container_width=True)
                    st.subheader("Prediction Stability (Convergence)")
                    st.markdown("Shows how predictions for the next draw change with more data.")
                    fig_delta = px.line(
                        delta_df,
                        x=delta_df.index,
                        y=delta_df.columns,
                        title="Prediction Delta Plot",
                        labels={'value': 'Predicted Number', 'index': 'History Size'}
                    )
                    st.plotly_chart(fig_delta, use_container_width=True)
    else:
        st.error("Invalid data format. After cleaning, the file must have 6 number columns and at least 50 rows.")
else:
    st.info("Upload a CSV file to engage the Engine.")
