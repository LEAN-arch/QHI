# ======================================================================================================
# LottoSphere v16.0.4: The Quantum Chronodynamics Engine
#
# AUTHOR: Subject Matter Expert AI (Stochastic Systems, Predictive Dynamics & Complex Systems)
# DATE: 2025-07-25
# VERSION: 16.0.4 (Debugged, Optimized with Periodicity Analysis, Position-Specific Max Numbers,
#                  and Substantially Expanded Explanation in UI)
#
# DESCRIPTION:
# A professional-grade scientific instrument for analyzing high-dimensional, chaotic time-series
# data, framed around lottery number sets. Models each of six sorted number positions as an
# independent yet interacting dynamical system. Integrates deep learning, statistical physics,
# chaos theory, and quantum-inspired methods with a robust metrology suite.
# Enhanced with periodicity analysis for non-positive Lyapunov exponents, position-specific
# maximum numbers based on sequence length (3 to 6), and a substantially expanded explanation
# of results and plots in the System Dynamics Explorer tab, providing deeper mathematical,
# technical, and actionable insights.
# ======================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
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
from statsmodels.tsa.stattools import acf

# --- Deep Learning (PyTorch) ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. APPLICATION CONFIGURATION ---
st.set_page_config(
    page_title="LottoSphere v16.0.4: Quantum Chronodynamics",
    page_icon="⚛️",
    layout="wide",
)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================================================
# ALL FUNCTION DEFINITIONS
# ====================================================================================================

# --- 2. CORE UTILITIES & DATA HANDLING ---
@st.cache_data
def load_data(uploaded_file, max_nums):
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        # Validate number range (1–max_nums[i], positive integers)
        valid_range = np.ones(df.shape, dtype=bool)
        for i, max_num in enumerate(max_nums):
            valid_range[:, i] = (df.iloc[:, i] >= 1) & (df.iloc[:, i] <= max_num) & (df.iloc[:, i] == df.iloc[:, i].astype(int))
        if not valid_range.all():
            st.session_state.data_warning = f"Invalid numbers detected (must be integers between 1 and respective max for each position). Discarding invalid rows."
            df = df[valid_range.all(axis=1)].reset_index(drop=True)
        
        # Check for duplicates within rows
        num_cols = df.shape[1]
        unique_counts = df.apply(lambda x: len(set(x)), axis=1)
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
        # Ensure sorted values respect position-specific maximums
        for i, max_num in enumerate(max_nums):
            sorted_values[:, i] = np.clip(sorted_values[:, i], 1, max_num)
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
        
        # Periodicity Analysis (for non-positive Lyapunov exponent)
        if results['lyapunov'] <= 0:
            try:
                # Compute autocorrelation function
                acf_vals = acf(series, nlags=min(50, len(series)//2), fft=True)
                lags = np.arange(len(acf_vals))
                
                # Find significant peaks (exclude lag 0)
                acf_peaks = acf_vals[1:]  # Skip lag 0
                lags_peaks = lags[1:]
                significant_peaks = acf_peaks > 0.2  # Threshold for significant correlation
                if np.any(significant_peaks):
                    dominant_period = lags_peaks[np.argmax(acf_peaks[significant_peaks])]
                    results['periodicity'] = dominant_period
                    results['periodicity_description'] = (
                        f"The system exhibits potential periodicity with a dominant period of approximately {dominant_period} draws. "
                        f"This suggests that number patterns may repeat every {dominant_period} draws, indicating a stable or cyclic behavior."
                    )
                else:
                    results['periodicity'] = None
                    results['periodicity_description'] = (
                        "No significant periodicity detected. The system is stable but may not exhibit clear repeating patterns."
                    )
                
                # Autocorrelation plot
                acf_df = pd.DataFrame({'Lag': lags, 'Autocorrelation': acf_vals})
                results['acf_fig'] = px.line(
                    acf_df,
                    x='Lag',
                    y='Autocorrelation',
                    title=f'Autocorrelation Function ({position})',
                    markers=True
                )
                results['acf_fig'].add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Significance Threshold")
                results['acf_fig'].add_hline(y=-0.2, line_dash="dash", line_color="red")
            except Exception as e:
                st.warning(f"Periodicity analysis failed: {e}")
                results['periodicity'] = None
                results['periodicity_description'] = "Periodicity analysis failed due to insufficient data or numerical issues."
                results['acf_fig'] = None

        return results
    except Exception as e:
        st.error(f"Error in temporal behavior analysis: {e}")
        return {}

# --- 4. ADVANCED PREDICTIVE MODELS ---

@st.cache_resource
def train_torch_model(_df, model_type='LSTM', seq_length=3, epochs=100, batch_size=32):
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
def predict_torch_model(_df, _model_cache, model_type='LSTM', seq_length=3, max_nums=[49]*6):
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
        prediction = np.clip(np.round(prediction), 1, max_nums).astype(int)
        unique_preds = []
        seen = set()
        for i, p in enumerate(prediction):
            candidate = p
            while candidate in seen:
                candidate = np.random.randint(1, max_nums[i] + 1)
            unique_preds.append(candidate)
            seen.add(candidate)
        
        # Fill with random valid numbers if needed
        while len(unique_preds) < 6:
            pos_idx = len(unique_preds)
            new_num = np.random.randint(1, max_nums[pos_idx] + 1)
            while new_num in seen:
                new_num = np.random.randint(1, max_nums[pos_idx] + 1)
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
def analyze_hilbert_embedding(_df, max_nums=[49]*6):
    try:
        if len(_df) < 2:
            raise ValueError("Insufficient data for Hilbert embedding")
        
        # Use maximum of position-specific maxs for complex plane
        max_num = max(max_nums)
        
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
        
        # Map selected numbers to position-specific bounds
        prediction = []
        seen = set()
        for i, num in enumerate(sorted(all_nums[selected])):
            pos_max = max_nums[i]
            candidate = min(num, pos_max)
            while candidate in seen:
                candidate = np.random.randint(1, pos_max + 1)
            prediction.append(candidate)
            seen.add(candidate)
        
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
def run_full_backtest_suite(df, max_nums=[49]*6):
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
            'LSTM': lambda d: predict_torch_model(d, lstm_cache, 'LSTM', max_nums=max_nums),
            'GRU': lambda d: predict_torch_model(d, gru_cache, 'GRU', max_nums=max_nums),
            'Hilbert Embedding': lambda d: analyze_hilbert_embedding(d, max_nums=max_nums),
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
def analyze_predictive_maturity(df, model_type='LSTM', max_nums=[49]*6):
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
            pred_obj = predict_torch_model(subset_df, model_cache, model_type, max_nums=max_nums)
            prediction_deltas.append(pred_obj['prediction'])
            
            split = max(40, int(len(subset_df) * 0.8))
            if len(subset_df) - split < 10:
                continue
            train, val = subset_df.iloc[:split], subset_df.iloc[split:]
            
            val_preds = [
                predict_torch_model(subset_df.iloc[:split+i], model_cache, model_type, max_nums=max_nums)['prediction']
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

# ====================================================================================================
# Main Application UI & Logic
# ====================================================================================================

st.title("⚛️ LottoSphere v16.0.4: The Quantum Chronodynamics Engine")
st.markdown("A scientific instrument for exploratory analysis of high-dimensional, chaotic systems. Models each number position as an evolving system using advanced mathematical and AI techniques.")

if 'data_warning' not in st.session_state:
    st.session_state.data_warning = None
if 'error_messages' not in st.session_state:
    st.session_state.error_messages = []

# Sidebar Configuration
st.sidebar.header("Configuration")
seq_length = st.sidebar.slider("Sequence Length", min_value=3, max_value=6, value=3)

# Default max numbers based on sequence length
default_max_num = 49 + (6 - seq_length) * 10
max_nums = []
for i in range(6):
    max_num = st.sidebar.number_input(
        f"Max Number Pos_{i+1}",
        min_value=10,
        max_value=100,
        value=min(default_max_num, 100),
        key=f"max_num_pos_{i+1}"
    )
    max_nums.append(max_num)

epochs = st.sidebar.slider("Training Epochs", min_value=50, max_value=200, value=100)

uploaded_file = st.sidebar.file_uploader("Upload Number.csv", type=["csv"])

if st.session_state.error_messages:
    for msg in st.session_state.error_messages:
        st.error(msg)
    st.session_state.error_messages = []

if uploaded_file:
    df_master = load_data(uploaded_file, max_nums)
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
                    scored_predictions = run_full_backtest_suite(df_master, max_nums=max_nums)
                
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
            
            # Add expanded explanation of results and plots
            with st.expander("Explanation of Results and Plots"):
                st.markdown("""
                ### Overview
                The System Dynamics Explorer analyzes the temporal behavior of a selected position (e.g., Pos_1, the smallest number in each sorted draw) as a dynamical system, leveraging principles from chaos theory, statistical physics, and time-series analysis. This module treats each position as an independent yet interacting component of a high-dimensional system, aiming to uncover patterns, cycles, or chaotic behavior in lottery number sequences. By modeling the sequence of numbers as a time series, it provides insights into whether the system is predictable (stable or periodic) or inherently unpredictable (chaotic), which is critical for developing informed number selection strategies or guiding further analytical efforts.

                The analysis produces four main outputs, each offering a unique perspective on the system’s dynamics:
                - **Recurrence Plot**: Visualizes when the time series revisits similar states, revealing repetitive patterns or chaotic scattering.
                - **Power Spectral Density (Fourier Analysis)**: Identifies dominant periodic components (cycles) in the time series, quantifying their strength.
                - **Continuous Wavelet Transform**: Maps how periodic patterns evolve over time, capturing non-stationary behavior.
                - **Lyapunov Exponent and Periodicity Analysis**: Quantifies the degree of chaos (or stability) and detects specific cycle lengths when the system is stable or periodic.

                These outputs collectively enable users to assess the predictability of the selected position’s numbers, distinguish between random and structured behavior, and develop actionable strategies for number selection. In the context of lotteries, where numbers are often assumed to be random, positional analysis (due to sorting) may reveal subtle patterns influenced by game mechanics or historical trends.

                ### Detailed Explanation of Each Result

                #### 1. Recurrence Plot
                **Description**:
                - The recurrence plot is a square matrix where each element \([i, j]\) represents the absolute difference between the time series values at draws \(i\) and \(j\), normalized to the range [0, 1]. Mathematically, for a time series \(x(t)\), the recurrence matrix is defined as \( R_{i,j} = |x_i - x_j| / \max(|x_i - x_j|) \), where \(x_i\) is the value at draw \(i\). Darker colors indicate smaller differences (similar states, or recurrences), while lighter colors indicate larger differences (dissimilar states).
                - The plot is visualized as a heatmap using a `viridis` color scale, with both axes representing draw indices (e.g., 0 to 999 for 1000 draws).
                - **Example**: For a time series of Pos_1 with 1000 draws, a 1000x1000 heatmap with dark diagonal lines every 10 draws suggests that similar numbers (e.g., Pos_1 = 3) reappear approximately every 10 draws.

                **Mathematical Foundation**:
                - Recurrence plots are rooted in state-space reconstruction, a technique from dynamical systems theory. The time series is embedded in a higher-dimensional space to capture its dynamics, but here, a simplified distance-based approach is used for computational efficiency.
                - The plot approximates the recurrence of states in the phase space, where \( R_{i,j} \approx 0 \) indicates that the system’s state at time \(i\) is close to its state at time \(j\).
                - Patterns in the plot reflect the system’s determinism: periodic systems show regular structures (e.g., diagonal lines), while chaotic systems exhibit scattered or noisy patterns.

                **Significance**:
                - **Patterns**: Diagonal lines parallel to the main diagonal (\(i = j\)) indicate periodic behavior, as the system returns to similar states at fixed intervals (e.g., every 10 draws). A checkerboard pattern suggests quasi-periodicity, where cycles vary slightly. A noisy, unstructured plot indicates chaotic or random behavior, with no consistent recurrences.
                - **In Lottery**: For Pos_1, the plot reveals whether specific numbers (e.g., 3, 5, 7) recur predictably. For instance, if dark lines appear every 10 draws, Pos_1 may cycle through a set of numbers, suggesting a periodic structure exploitable for predictions.
                - **Limitations**:
                  - **Noise Sensitivity**: Lottery data often contains random fluctuations, which can obscure patterns, especially in short datasets (<100 draws).
                  - **Data Requirements**: Requires at least 50 draws (enforced by `load_data`) to capture meaningful recurrences. Longer series (>500 draws) improve pattern clarity.
                  - **Interpretation**: Patterns may be subtle, requiring cross-validation with other analyses (e.g., Periodicity Analysis) to confirm periodicity.

                **Actionability**:
                - **Periodic Behavior**: If diagonal lines appear every 10 draws, hypothesize a 10-draw cycle. For example, if Pos_1 was 5 ten draws ago, prioritize 5 for the next prediction. Review recent draws to identify numbers within the cycle.
                - **Quasi-Periodic Behavior**: A checkerboard pattern suggests variable cycles. Consider a range of lags (e.g., 8–12 draws) and select numbers that appeared in those draws.
                - **Chaotic Behavior**: A noisy plot with no clear structure indicates unpredictability. Shift focus to the Predictive Analytics tab (Tab 1), where LSTM, GRU, or Hilbert Embedding models can capture complex, non-linear patterns.
                - **Practical Steps**:
                  - Extract cycle length from diagonal line spacing (e.g., 10 draws).
                  - Use `df_master['Pos_1'].iloc[-10]` to check the number 10 draws ago and include it in your prediction set.
                  - If patterns are unclear, collect more historical draws (e.g., >1000) to enhance recurrence visibility or adjust the normalization factor in the code for finer resolution.
                - **Cross-Validation**: Confirm cycle lengths with Periodicity Analysis (ACF) and Fourier Analysis. If inconsistent, the system may be chaotic or data-limited.

                #### 2. Power Spectral Density (Fourier Analysis)
                **Description**:
                - The Power Spectral Density (PSD) plot estimates the power (strength) of periodic components in the time series across different frequencies, computed using Welch’s method. The x-axis represents frequency in cycles per draw (e.g., 0.1 cycles/draw), and the y-axis shows power (amplitude squared of the frequency component). Peaks indicate dominant cycles.
                - Welch’s method divides the time series into overlapping segments, applies a Fourier transform to each, and averages the results to reduce noise, with segment length set to `min(len(series), 256)` draws.
                - **Example**: A peak at 0.1 cycles/draw corresponds to a period of \(1/0.1 = 10\) draws, suggesting Pos_1 numbers may repeat every 10 draws.

                **Mathematical Foundation**:
                - The PSD is computed as \( P(f) = \frac{1}{N} |\sum_{t=0}^{N-1} x(t) e^{-i2\pi ft}|^2 \), averaged over segments, where \(x(t)\) is the time series, \(f\) is frequency, and \(N\) is the segment length.
                - Frequency \(f\) in cycles/draw relates to period \(T\) via \(T = 1/f\). For example, \(f = 0.05\) implies \(T = 20\) draws.
                - Welch’s method improves robustness by reducing variance through overlapping windows, but assumes local stationarity within segments.

                **Significance**:
                - **Frequencies**: Peaks in the PSD indicate dominant periodic components. A strong peak at 0.05 cycles/draw suggests a 20-draw cycle, meaning Pos_1 numbers (e.g., 5, 7) may recur every 20 draws.
                - **In Lottery**: Identifies cycle lengths that can guide number selection. For example, a 10-draw cycle suggests selecting numbers from 10 draws ago, as they may reappear.
                - **Limitations**:
                  - **Stationarity Assumption**: Assumes consistent statistical properties, which lottery data may violate due to randomness or external factors (e.g., rule changes).
                  - **Short Datasets**: With <100 draws, PSD peaks may be noisy or unreliable due to limited frequency resolution.
                  - **Resolution**: The segment length (`nperseg=256`) limits the ability to detect long cycles (>256 draws) unless the dataset is sufficiently large.

                **Actionability**:
                - **Cycle Detection**:
                  - Identify the top 2–3 peaks in the PSD plot. Compute periods as \(T = 1/f\). For a peak at 0.1 cycles/draw, expect a 10-draw cycle.
                  - For Pos_1, check `df_master['Pos_1'].iloc[-T]` to select candidate numbers (e.g., if Pos_1 was 3 ten draws ago, include 3).
                - **Validation**:
                  - Compare PSD periods with Periodicity Analysis (ACF) and Wavelet Transform bands to confirm cycle consistency.
                  - If no clear peaks appear, the system may be non-periodic or data-limited, suggesting reliance on Tab 1 models (LSTM/GRU).
                - **Practical Steps**:
                  - For a 10-draw cycle, review the last 10 draws for Pos_1 and prioritize recurring numbers.
                  - If multiple peaks exist (e.g., at 0.1 and 0.2 cycles/draw), test predictions for both 10- and 5-draw cycles to diversify candidates.
                  - If peaks are weak, increase `nperseg` (e.g., to 512) in `analyze_temporal_behavior` for finer resolution, provided the dataset has >512 draws.
                - **Optimization**:
                  - For large datasets, adjust `nperseg` to balance resolution and computation time (e.g., `nperseg=len(series)//2`).
                  - If results are noisy, apply a smoothing filter (e.g., moving average) to the time series before analysis, though this requires code modification.

                #### 3. Continuous Wavelet Transform
                **Description**:
                - The Continuous Wavelet Transform (CWT) decomposes the time series into time-frequency components using a Morlet wavelet, which balances time and frequency resolution. The x-axis represents time (draw indices), the y-axis represents scales (inversely related to frequency), and color intensity reflects the amplitude of wavelet coefficients.
                - Scales are set to `np.arange(1, min(31, len(series)//2))`, limiting analysis to short-to-medium cycles due to computational constraints.
                - **Example**: A bright band at scale 10 around draw 500 indicates a strong periodic signal with a period of approximately 10 draws at that time.

                **Mathematical Foundation**:
                - The CWT is defined as \( W(s, t) = \int x(\tau) \psi^*((t - \tau)/s) d\tau \), where \(x(t)\) is the time series, \(\psi\) is the Morlet wavelet, \(s\) is the scale, and \(t\) is time.
                - The Morlet wavelet, \(\psi(t) = \pi^{-1/4} e^{i\omega_0 t} e^{-t^2/2}\), is a complex wave modulated by a Gaussian, with \(\omega_0 = 6\) for balance between time and frequency localization.
                - Scale \(s\) relates to period via \( T \approx s \cdot \delta t \), where \(\delta t\) is the sampling interval (1 draw). For lottery data, \( T \approx s \).

                **Significance**:
                - **Time-Frequency Insight**: Unlike Fourier analysis, CWT captures non-stationary behavior, showing when cycles emerge, persist, or vanish. For example, a bright band at scale 10 from draws 200–300 indicates a 10-draw cycle active in that window.
                - **In Lottery**: Identifies periods when Pos_1 numbers follow cyclic patterns, enabling time-specific predictions. For instance, if a cycle is active in recent draws, numbers from the corresponding lag are strong candidates.
                - **Limitations**:
                  - **Computational Cost**: CWT is intensive for large datasets (>1000 draws), as the complexity scales with series length and number of scales.
                  - **Short Series**: With <100 draws, results may be noisy, missing long cycles.
                  - **Wavelet Choice**: The Morlet wavelet assumes smooth oscillations, potentially missing abrupt changes in lottery data.

                **Actionability**:
                - **Cycle Timing**:
                  - Identify bright bands and their scales (e.g., scale 10 ≈ 10 draws). If a band appears in recent draws (e.g., last 50 draws), prioritize numbers from the corresponding lag (e.g., `df_master['Pos_1'].iloc[-10]`).
                  - For example, if Pos_1 = 7 ten draws ago and a band is active at scale 10, include 7 in predictions.
                - **Dynamic Adjustments**:
                  - If cycles are time-specific (e.g., draws 200–300), focus on recent data for predictions, as older cycles may no longer apply.
                  - If no bands appear, the system may lack strong periodicity, suggesting reliance on Tab 1 models.
                - **Practical Steps**:
                  - Cross-check scale-based periods with PSD and ACF results. For a scale 10 band, confirm a PSD peak at 0.1 cycles/draw or an ACF peak at lag 10.
                  - If results are noisy, reduce the scale range (e.g., `widths=np.arange(1, 16)`) to focus on shorter cycles or increase dataset size.
                - **Optimization**:
                  - For large datasets, limit `widths` to 1–15 to reduce computation time without losing short-cycle detection.
                  - Consider alternative wavelets (e.g., Mexican Hat) for sharper transitions, though this requires code changes.

                #### 4. Lyapunov Exponent and Periodicity Analysis
                **Description**:
                - **Lyapunov Exponent**: Quantifies the rate of divergence of nearby trajectories in the time series, computed using Rosenstein’s algorithm (`nolds.lyap_r`) with an embedding dimension of 2. A positive exponent indicates chaotic behavior (exponential divergence), while a non-positive exponent (≤0) suggests stability or periodicity.
                - **Periodicity Analysis**: For non-positive Lyapunov exponents, the autocorrelation function (ACF) is computed to detect periodic patterns. The ACF measures correlation between the time series and its lagged versions, with lags up to `min(50, len(series)//2)`. Peaks above a threshold (0.2) indicate potential periods.
                - **Example**:
                  - Lyapunov = -0.0123 (stable), ACF peak at lag 8, with description: “The system exhibits potential periodicity with a dominant period of approximately 8 draws.”
                  - ACF plot: A line graph with x-axis as lags (0–50), y-axis as autocorrelation (-1 to 1), and red dashed lines at ±0.2 indicating significance.

                **Mathematical Foundation**:
                - **Lyapunov Exponent**: For a time series \(x(t)\), the largest Lyapunov exponent \(\lambda\) is estimated as the average divergence rate: \( d(t) \approx d_0 e^{\lambda t} \), where \(d(t)\) is the distance between nearby trajectories. The algorithm reconstructs the state space with embedding dimension 2, tracking divergence over time.
                - **ACF**: The autocorrelation at lag \(k\) is \( \rho(k) = \frac{\sum_{t=1}^{N-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^N (x_t - \bar{x})^2} \), where \(\bar{x}\) is the mean. Peaks above 0.2 indicate significant periodicity at lag \(k\).
                - The FFT-based ACF computation (`fft=True`) ensures efficiency for large datasets.

                **Significance**:
                - **Lyapunov Exponent**:
                  - **Positive (\(\lambda > 0\))**: Indicates chaos, where small changes in initial conditions lead to large prediction errors. For Pos_1, this suggests numbers are highly unpredictable, requiring robust models like LSTM/GRU.
                  - **Non-positive (\(\lambda \leq 0\))**: Suggests a stable or periodic system, where patterns may repeat predictably, enabling cycle-based predictions.
                - **Periodicity Analysis**:
                  - A peak at lag 8 indicates an 8-draw cycle, meaning Pos_1 numbers (e.g., 5) may repeat every 8 draws.
                  - No significant peaks suggest stability without clear cycles, implying a steady but non-repeating pattern.
                - **In Lottery**: A non-positive Lyapunov with a clear period (e.g., 8 draws) allows predictions based on historical values at that lag. Chaotic systems require diversified, model-based approaches.

                **Actionability**:
                - **Chaotic System (Positive Lyapunov)**:
                  - **Action**: Avoid cycle-based predictions due to sensitivity to initial conditions. Use Tab 1 models (LSTM, GRU, Hilbert Embedding) for robust forecasts, as they capture non-linear patterns.
                  - **Practical Steps**: Run the Predictive Analytics tab and select the highest-likelihood model (based on backtesting scores).
                  - **Next Steps**: Focus on Tab 1 results and monitor prediction accuracy in Tab 3 (Predictive Maturity) to assess model reliability.
                - **Stable/Periodic System (Non-positive Lyapunov)**:
                  - **If Period Detected**:
                    - For an ACF peak at lag 8, check `df_master['Pos_1'].iloc[-8]` (e.g., if Pos_1 = 5, consider 5 for the next draw).
                    - Build a candidate set from numbers at lags 8, 16, 24, etc., to capture the cycle.
                  - **No Period**:
                    - Perform frequency analysis: `df_master['Pos_1'].value_counts()` to identify the most common numbers (e.g., if 3 appears most often, prioritize 3).
                    - Combine with Tab 1 predictions for a diversified approach.
                  - **Practical Steps**:
                    - For a 10-draw cycle, review `df_master['Pos_1'].iloc[-10::10]` to extract recurring numbers.
                    - If no periodicity, use historical frequencies or Tab 1 models.
                - **ACF Plot Interpretation**:
                  - Peaks above the red dashed line (0.2) indicate significant periodicity. Use the lag (e.g., 8) to select numbers from that draw.
                  - Multiple peaks (e.g., at lags 8 and 16) suggest harmonics or multiple cycles; test both lags.
                  - If peaks are below 0.2, the system is stable but non-periodic; rely on frequency analysis or Tab 1.
                - **Validation**:
                  - Confirm ACF periods with PSD (peak at \(1/\text{period}\)) and CWT (band at scale ≈ period).
                  - If inconsistent, collect more data or adjust the ACF threshold (e.g., to 0.3 for stricter periodicity).
                - **Limitations**:
                  - **Lyapunov**: Sensitive to noise and short series (<100 draws), potentially yielding unreliable estimates. The embedding dimension (2) may be insufficient for complex dynamics.
                  - **ACF**: Assumes stationarity, which lottery data may violate. Short lags (≤50) limit detection of long cycles.
                  - **Mitigation**: Use multiple analyses (PSD, CWT) for confirmation. Increase data length or adjust parameters (e.g., `nlags=100`) for longer cycles.

                ### Integrated Interpretation
                **Combining Results**:
                - **Coherent Patterns**: If the Recurrence Plot shows diagonal lines every 10 draws, PSD shows a peak at 0.1 cycles/draw, CWT shows a bright band at scale 10, and Periodicity Analysis (for non-positive Lyapunov) confirms a 10-draw cycle, Pos_1 is strongly periodic. This suggests a robust cycle for predictions, e.g., selecting numbers from 10 draws ago.
                - **Chaotic Signals**: A noisy Recurrence Plot, flat PSD, scattered CWT, and positive Lyapunov indicate chaos. Predictions should rely on Tab 1 models, which are designed for complex, non-linear systems.
                - **Mixed Signals**: A non-positive Lyapunov with no ACF peaks may indicate stability without periodicity, possibly due to insufficient data or weak cycles. A noisy Recurrence Plot with weak PSD/CWT signals suggests randomness or data limitations.
                - **Error Analysis**: Discrepancies between analyses (e.g., PSD peak but no ACF peak) may arise from noise, non-stationarity, or short datasets. Collect more draws (>500) or adjust parameters (e.g., increase ACF lags, reduce CWT scales).

                **Significance in Lottery**:
                - Lottery systems are often assumed to be random, but positional analysis (e.g., Pos_1 as the smallest number) can reveal patterns due to sorting constraints or game mechanics (e.g., maximum numbers per position). The System Dynamics Explorer quantifies these patterns, distinguishing between random, periodic, or chaotic behavior.
                - Periodic systems enable cycle-based predictions, while chaotic systems require robust statistical or machine learning models. Stable but non-periodic systems suggest steady but unpredictable patterns, favoring frequency-based approaches.

                **Actionable Strategy**:
                1. **Check Lyapunov Exponent**:
                   - **Positive**: Indicates chaos. Use Tab 1 predictions (LSTM/GRU/Hilbert) and avoid cycle-based strategies.
                   - **Non-positive**: Proceed to Periodicity Analysis, PSD, and CWT to explore cyclic or stable behavior.
                2. **Analyze Periodicity**:
                   - If ACF shows a peak (e.g., lag 8), select Pos_1 numbers from 8 draws ago (e.g., `df_master['Pos_1'].iloc[-8]`).
                   - If no peaks, compute frequency counts (`df_master['Pos_1'].value_counts()`) and prioritize common numbers.
                3. **Cross-Validate**:
                   - Confirm ACF periods with PSD (e.g., peak at 0.125 cycles/draw for 8 draws) and CWT (band at scale ≈ 8).
                   - Check Recurrence Plot for diagonal lines matching the period.
                   - If inconsistent, increase data length or adjust parameters (e.g., ACF threshold to 0.3, PSD `nperseg` to 512).
                4. **Integrate with Predictions**:
                   - Combine cycle-based candidates (e.g., numbers from 8 draws ago) with Tab 1 predictions to create a diversified set.
                   - Use Tab 3 (Predictive Maturity) to assess whether more data improves cycle reliability or model performance.
                5. **Iterate and Validate**:
                   - Test cycle-based predictions against new draws. If accurate, continue using the cycle; if not, reassess with more data or shift to Tab 1 models.
                   - Monitor Tab 3 for convergence in predictions, indicating robust cycles or model stability.

                ### Technical Notes
                - **Data Requirements**:
                  - Minimum 50 draws (enforced by `load_data`) for basic analysis. Longer series (>100 for ACF, >500 for PSD/CWT) improve reliability.
                  - Short datasets (<100 draws) may yield noisy or inconclusive results, especially for Lyapunov and CWT.
                - **Parameter Tuning**:
                  - **ACF Threshold**: The 0.2 threshold balances sensitivity and specificity. Increase to 0.3 for stricter periodicity or decrease to 0.15 for weak signals, adjusting in `analyze_temporal_behavior`.
                  - **PSD Resolution**: Increase `nperseg` (e.g., to 512) for finer frequency resolution if data length allows (>512 draws).
                  - **CWT Scales**: Reduce `widths` (e.g., to 1–15) for faster computation or focus on short cycles. Increase for long datasets to detect longer periods.
                - **Performance Optimization**:
                  - Large datasets (>1000 draws) slow Recurrence Plots and CWT. Subsample data (e.g., every 2nd draw) or reduce `widths` to 1–15.
                  - Cache results with `@st.cache_data` to avoid recomputation unless data or parameters change.
                - **Robustness**:
                  - Error handling ensures graceful failure (e.g., Lyapunov or ACF failures display warnings).
                  - Check `st.session_state.data_warning` for data issues (e.g., invalid numbers, duplicates).
                - **Enhancements**:
                  - Add interactive plot controls (e.g., zoom, hover for values) by modifying Plotly configurations.
                  - Implement alternative wavelets or higher embedding dimensions for Lyapunov analysis if needed, requiring code changes.
                """)

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
                        if 'periodicity_description' in dynamic_results:
                            st.info(
                                f"**Periodicity Analysis:** {dynamic_results['periodicity_description']}",
                                icon="🔄"
                            )
                            if dynamic_results['acf_fig'] is not None:
                                st.plotly_chart(dynamic_results['acf_fig'], use_container_width=True)

        with tab3:
            st.header("Predictive Maturity Analysis")
            st.markdown("Determine how predictive power evolves with historical data size.")
            model_type = st.selectbox("Select Model", options=['LSTM', 'GRU'], index=0)
            if st.button("RUN MATURITY ANALYSIS"):
                with st.spinner("Performing iterative backtesting... This is computationally expensive."):
                    maturity_df, delta_df = analyze_predictive_maturity(df_master, model_type=model_type, max_nums=max_nums)
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
