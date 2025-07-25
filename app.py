# ======================================================================================================
# LottoSphere v16.0.3: The Quantum Chronodynamics Engine
#
# AUTHOR: Subject Matter Expert AI (Stochastic Systems, Predictive Dynamics & Complex Systems)
# DATE: 2025-07-25
# VERSION: 16.0.3 (Debugged, Optimized with Periodicity Analysis, Position-Specific Max Numbers,
#                  and Detailed Explanation in UI)
#
# DESCRIPTION:
# A professional-grade scientific instrument for analyzing high-dimensional, chaotic time-series
# data, framed around lottery number sets. Models each of six sorted number positions as an
# independent yet interacting dynamical system. Integrates deep learning, statistical physics,
# chaos theory, and quantum-inspired methods with a robust metrology suite.
# Enhanced with periodicity analysis for non-positive Lyapunov exponents, position-specific
# maximum numbers based on sequence length (3 to 6), and a detailed explanation of results
# and plots in the System Dynamics Explorer tab.
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
    page_title="LottoSphere v16.0.3: Quantum Chronodynamics",
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

st.title("⚛️ LottoSphere v16.0.3: The Quantum Chronodynamics Engine")
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
            
            # Add detailed explanation of results and plots
            with st.expander("Explanation of Results and Plots"):
                st.markdown("""
                ### Overview
                The System Dynamics Explorer analyzes the temporal behavior of a selected position (e.g., Pos_1, the smallest number in each draw) as a dynamical system. It provides insights into whether the time series is chaotic, stable, or periodic, which is critical for understanding the underlying dynamics of the lottery system. The analysis includes four main outputs:

                - **Recurrence Plot**: Visualizes when the time series revisits similar states.
                - **Power Spectral Density (Fourier Analysis)**: Identifies dominant frequencies (cycles) in the time series.
                - **Continuous Wavelet Transform**: Reveals how frequencies evolve over time.
                - **Lyapunov Exponent and Periodicity Analysis**: Quantifies chaos (or stability) and detects periodic cycles when the system is stable or periodic.

                These outputs help assess whether the position’s numbers are predictable, cyclic, or chaotic, informing strategies for number selection or further analysis.

                ### Detailed Explanation of Each Result

                #### 1. Recurrence Plot
                **Description**:
                - A square heatmap where each point \([i, j]\) shows the absolute difference between the values at draws \(i\) and \(j\), normalized to [0, 1]. Darker colors indicate similar values (recurrence of states), while lighter colors show dissimilar states.
                - **Example**: For 1000 draws, a 1000x1000 heatmap with dark diagonal lines suggests repeated patterns.

                **Significance**:
                - **Patterns**: Diagonal lines parallel to the main diagonal (where \(i = j\)) indicate periodic behavior, as the same numbers reappear at regular intervals. A checkerboard pattern suggests quasi-periodicity, while a noisy plot indicates chaotic or random behavior.
                - **In Lottery**: Reveals whether numbers in the selected position (e.g., Pos_1 = 3) recur predictably. Long diagonal lines suggest a cycle, useful for anticipating reappearances.
                - **Limitations**: Sensitive to noise in lottery data, which may obscure patterns. Requires sufficient data (≥50 draws) for meaningful recurrences.

                **Actionability**:
                - **Periodic Behavior**: If diagonal lines appear every 10 draws, hypothesize a 10-draw cycle. Prioritize numbers that appeared 10 draws ago for Pos_1.
                - **Chaotic Behavior**: A noisy plot suggests unpredictability. Rely on Predictive Analytics (Tab 1) models like LSTM/GRU instead of cycle-based strategies.
                - **Next Steps**: Confirm cycles with Periodicity Analysis. Collect more data if patterns are unclear.

                #### 2. Power Spectral Density (Fourier Analysis)
                **Description**:
                - A line plot showing the power (strength) of different frequencies in the time series. The x-axis is frequency (cycles per draw), and the y-axis is power. Peaks indicate dominant cycles.
                - **Example**: A peak at 0.1 cycles/draw suggests a cycle every \(1/0.1 = 10\) draws.

                **Significance**:
                - **Frequencies**: Peaks show cycle lengths. A peak at 0.05 cycles/draw means a 20-draw cycle, suggesting Pos_1 numbers may repeat every 20 draws.
                - **In Lottery**: Helps identify cyclic patterns for number selection based on cycle timing.
                - **Limitations**: Assumes stationarity, which lottery data may violate due to randomness. Short datasets (<100 draws) may produce noisy results.

                **Actionability**:
                - **Cycle Detection**: Note the top 2–3 peaks and compute periods (period = \(1/\text{frequency}\)). For a peak at 0.1, expect a 10-draw cycle. Select numbers from 10 draws ago.
                - **Validation**: Compare with Periodicity Analysis and Wavelet Transform. If no peaks, rely on Tab 1 models.
                - **Next Steps**: Analyze recent draws to align predictions with the cycle. Adjust analysis parameters (e.g., increase frequency resolution) for clearer peaks if data allows.

                #### 3. Continuous Wavelet Transform
                **Description**:
                - A heatmap showing time-frequency patterns. The x-axis is time (draw indices), the y-axis is scale (inversely related to frequency), and color intensity shows the strength of periodic signals.
                - **Example**: A bright band at scale 10 around draw 500 indicates a strong 10-draw cycle at that time.

                **Significance**:
                - **Time-Frequency Insight**: Unlike Fourier, it captures non-stationary behavior, showing when cycles appear or vanish.
                - **In Lottery**: Bright bands indicate periods when Pos_1 numbers recur cyclically, guiding predictions based on active cycles.
                - **Limitations**: Computationally intensive for large datasets (>1000 draws). Noisy for short series. Assumes smooth oscillations, which may miss abrupt changes.

                **Actionability**:
                - **Cycle Timing**: Identify bright bands and their scales (e.g., scale 10 ≈ 10 draws). If recent draws are in a bright region, prioritize numbers from recent cycles.
                - **Dynamic Adjustments**: Focus on recent data if cycles are time-specific (e.g., draws 200–300).
                - **Next Steps**: Correlate with Fourier and Periodicity Analysis. Adjust scale range for faster computation if needed.

                #### 4. Lyapunov Exponent and Periodicity Analysis
                **Description**:
                - **Lyapunov Exponent**: Measures chaos. Positive values indicate chaotic behavior (sensitivity to initial conditions); non-positive (≤0) suggest stability or periodicity.
                - **Periodicity Analysis**: For non-positive Lyapunov, an autocorrelation function (ACF) plot shows correlations at different lags. Peaks above 0.2 indicate periods (e.g., a peak at lag 8 suggests an 8-draw cycle).
                - **Example**: Lyapunov = -0.0123 (stable), ACF peak at lag 8 with description: “Periodicity with a dominant period of approximately 8 draws.”

                **Significance**:
                - **Lyapunov**:
                  - **Positive**: Chaotic system, unpredictable. Use robust models (Tab 1).
                  - **Non-positive**: Stable or periodic, allowing cycle-based predictions.
                - **Periodicity**:
                  - A peak at lag 8 indicates an 8-draw cycle for Pos_1 numbers.
                  - No peaks suggest stability without clear cycles.
                - **In Lottery**: A stable system with a period (e.g., 8 draws) enables predictions based on cycle timing. Chaotic systems require diversified models.

                **Actionability**:
                - **Chaotic (Positive Lyapunov)**:
                  - Avoid cycle-based predictions. Use Tab 1 (LSTM/GRU/Hilbert) for robust forecasts.
                  - Next: Focus on Predictive Analytics.
                - **Stable/Periodic (Non-positive Lyapunov)**:
                  - If period detected (e.g., 8 draws), select Pos_1 numbers from 8 draws ago. If Pos_1 was 5 then, consider 5.
                  - No period: Use frequency analysis (e.g., most common Pos_1 numbers).
                  - Next: Confirm period with Fourier/Wavelet. Test cycle-based predictions in Tab 1.
                - **ACF Plot**:
                  - Peaks above red lines (0.2) indicate periodicity. Use lag (e.g., 8) to predict based on past draws.
                  - No peaks: Rely on other analyses or Tab 1.
                - **Limitations**: Lyapunov is noise-sensitive; ACF assumes stationarity. Use multiple analyses for confirmation.

                ### Integrated Interpretation
                **Combining Results**:
                - **Coherent Patterns**: If Recurrence shows diagonal lines, Fourier shows a peak (e.g., 0.1 cycles/draw), Wavelet shows a band at scale 10, and Periodicity confirms a 10-draw cycle, Pos_1 is periodic. Predict based on 10-draw cycles.
                - **Chaotic Signals**: Noisy Recurrence, flat Fourier, scattered Wavelet, and positive Lyapunov suggest chaos. Use Tab 1 models.
                - **Mixed Signals**: Negative Lyapunov but no ACF peaks may indicate stability without periodicity or insufficient data. Collect more draws.

                **Significance in Lottery**:
                - Positional analysis (e.g., Pos_1) can reveal patterns due to sorting. This tab quantifies predictability, guiding cycle-based or model-based strategies.

                **Actionable Strategy**:
                1. **Check Lyapunov**:
                   - Positive: Use Tab 1 predictions.
                   - Non-positive: Proceed to Periodicity and cross-check with Fourier/Wavelet.
                2. **Analyze Periodicity**:
                   - Use ACF to identify period (e.g., 8 draws). Select numbers from 8 draws ago.
                   - No period: Prioritize frequent Pos_1 numbers.
                3. **Cross-Validate**:
                   - Confirm period with Fourier (peak at \(1/\text{period}\)) and Wavelet (band at scale ≈ period).
                   - Check Recurrence for cycle consistency.
                4. **Integrate**:
                   - Combine cycle candidates with Tab 1 predictions.
                   - Use Tab 3 to validate cycle reliability with more data.
                5. **Iterate**:
                   - Test predictions against new draws. Adjust strategy if cycles fail.

                ### Technical Notes
                - **Data**: Requires ≥50 draws. Longer series (>100) improve Periodicity reliability.
                - **Parameters**: Adjust ACF threshold (0.2), Fourier resolution, or Wavelet scales if needed.
                - **Performance**: Large datasets (>1000 draws) may slow Recurrence/Wavelet. Optimize by reducing scales or subsampling.
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
