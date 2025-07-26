# ======================================================================================================
# LottoSphere v16.0.15: The Quantum Chronodynamics Engine (Final)
#
# AUTHOR: Subject Matter Expert AI (Stochastic Systems, Predictive Dynamics & Complex Systems)
# DATE: 2025-07-26
# VERSION: 16.0.15 (Final)
#
# DESCRIPTION:
# A professional-grade scientific instrument for analyzing high-dimensional, chaotic time-series
# data, framed around lottery number sets. This version is stabilized by removing the problematic
# Bayesian Neural Network model to resolve deep dependency conflicts.
#
# CHANGELOG (from v16.0.14 to v16.0.15):
# - FIXED: Corrected an AttributeError in the HMM model by updating the emission probability
#   attribute from the deprecated 'emissionprob_' to the correct 'emission_prob_'.
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
from typing import List, Dict, Any, Tuple, Optional

# --- Suppress Warnings for a Cleaner UI ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Advanced Scientific & ML Libraries ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.signal import welch, spectrogram
from nolds import lyap_r
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from prophet import Prophet
from hmmlearn.hmm import MultinomialHMM
from sktime.forecasting.arima import AutoARIMA

# --- Deep Learning (PyTorch) ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. APPLICATION CONFIGURATION & INITIALIZATION ---
st.set_page_config(
    page_title="LottoSphere v16.0.15: Quantum Chronodynamics",
    page_icon="⚛️",
    layout="wide",
)
# Initialize session state keys
if 'data_warnings' not in st.session_state:
    st.session_state.data_warnings = []

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================================================
# ALL FUNCTION DEFINITIONS
# ====================================================================================================

# --- 2. CORE UTILITIES & DATA HANDLING ---
@st.cache_data
def load_data(uploaded_file: io.BytesIO, max_nums: List[int]) -> pd.DataFrame:
    """
    Loads, validates, and preprocesses the lottery data from a CSV file.
    """
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        st.session_state.data_warnings = []

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        df = df.astype(int)

        initial_rows = len(df)
        for i, max_num in enumerate(max_nums):
            if i < df.shape[1]:
                df = df[(df.iloc[:, i] >= 1) & (df.iloc[:, i] <= max_num)]
        if len(df) < initial_rows:
            st.session_state.data_warnings.append(
                f"Discarded {initial_rows - len(df)} rows with numbers outside the specified max range."
            )

        num_cols = min(6, df.shape[1])
        df = df.iloc[:, :num_cols]
        unique_counts = df.apply(lambda x: len(set(x)), axis=1)
        valid_rows_mask = (unique_counts == num_cols)
        if not valid_rows_mask.all():
            st.session_state.data_warnings.append(
                f"Discarded {len(df) - valid_rows_mask.sum()} rows with duplicate numbers within the draw."
            )
            df = df[valid_rows_mask]

        if df.duplicated().any():
            st.session_state.data_warnings.append(f"Discarded {df.duplicated().sum()} duplicate rows.")
            df = df.drop_duplicates()

        df = df.reset_index(drop=True)
        df.columns = [f'Pos_{i+1}' for i in range(df.shape[1])]

        if len(df) < 50:
            st.session_state.data_warnings.append("Insufficient data for robust analysis (at least 50 rows required).")
            return pd.DataFrame()

        st.session_state.data_warnings.append("Input data sorted per row to create positional time series. Last rows are treated as most recent draws.")
        sorted_values = np.sort(df.values, axis=1)
        return pd.DataFrame(sorted_values, columns=df.columns).astype(int)

    except Exception as e:
        st.error(f"Fatal error loading data: {e}")
        return pd.DataFrame()

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates overlapping sequences for time-series model training."""
    if seq_length >= len(data):
        raise ValueError("Sequence length must be less than data length")
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 3. TIME-DEPENDENT BEHAVIOR ANALYSIS MODULE ---
@st.cache_data
def analyze_temporal_behavior(_df: pd.DataFrame, position: str = 'Pos_1') -> Dict[str, Any]:
    """
    Analyzes the chaotic and cyclical nature of a single positional time series.
    """
    try:
        results = {}
        series = _df[position].values

        recurrence_matrix = np.abs(np.subtract.outer(series, series))
        normalized_recurrence = recurrence_matrix / (recurrence_matrix.max() + 1e-10)
        results['recurrence_fig'] = px.imshow(normalized_recurrence, color_continuous_scale='viridis', title=f"Recurrence Plot ({position})")

        freqs, psd = welch(series, nperseg=min(len(series), 256))
        psd_df = pd.DataFrame({'Frequency': freqs, 'Power': psd}).sort_values('Power', ascending=False)
        results['fourier_fig'] = px.line(psd_df, x='Frequency', y='Power', title=f"Power Spectral Density ({position})")

        f, t, Sxx = spectrogram(series, fs=1.0, nperseg=min(128, len(series)//2), noverlap=int(min(128, len(series)//2)*0.9))
        results['spectrogram_fig'] = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx + 1e-10), x=t, y=f, colorscale='viridis'))
        results['spectrogram_fig'].update_layout(title=f'Spectrogram ({position})', xaxis_title='Time', yaxis_title='Frequency')

        try:
            lyap_exp = lyap_r(series, emb_dim=3, lag=1, min_tsep=len(series)//10)
            results['lyapunov'] = lyap_exp
            results['is_stable'] = (lyap_exp <= 0.05)
        except Exception:
            results['lyapunov'] = -1
            results['is_stable'] = True

        if results['is_stable']:
            acf_vals = acf(series, nlags=min(50, len(series)//2 - 1), fft=True)
            lags = np.arange(len(acf_vals))
            conf_interval = 1.96 / np.sqrt(len(series))
            significant_peaks_indices = np.where(acf_vals[1:] > conf_interval)[0]
            if significant_peaks_indices.size > 0:
                dominant_period = lags[1:][significant_peaks_indices[0]]
                results['periodicity'] = dominant_period
                results['periodicity_description'] = f"Potential periodicity with a dominant period of {dominant_period} draws."
            else:
                results['periodicity'] = None
                results['periodicity_description'] = "No significant periodicity detected."

            acf_df = pd.DataFrame({'Lag': lags, 'Autocorrelation': acf_vals})
            results['acf_fig'] = px.line(acf_df, x='Lag', y='Autocorrelation', title=f'Autocorrelation Function ({position})', markers=True)
            results['acf_fig'].add_hline(y=conf_interval, line_dash="dash", line_color="red")
            results['acf_fig'].add_hline(y=-conf_interval, line_dash="dash", line_color="red")

        return results
    except Exception as e:
        st.error(f"Error in temporal behavior analysis: {e}")
        return {}


# --- 4. ADVANCED STABLE POSITION ANALYSIS (REFACTORED) ---

def _analyze_stat_physics(series: np.ndarray, max_num: int) -> Dict[str, Any]:
    """Sub-module for Statistical Physics and Complex Systems analysis."""
    results = {}
    counts = np.zeros((max_num, max_num))
    for i in range(len(series)-1):
        from_idx, to_idx = series[i]-1, series[i+1]-1
        if 0 <= from_idx < max_num and 0 <= to_idx < max_num:
            counts[from_idx, to_idx] += 1
            
    trans_prob = (counts + 1e-10) / (counts.sum(axis=1, keepdims=True) + 1e-9)
    current_state = series[-1] - 1
    if not (0 <= current_state < max_num):
        current_state = np.random.randint(0, max_num)
        
    prob_vector = trans_prob[current_state]
    prob_vector /= prob_vector.sum()
    
    mcmc_samples = [np.random.choice(max_num, p=prob_vector) for _ in range(2000)]
    mcmc_dist = pd.Series(c + 1 for c in mcmc_samples).value_counts(normalize=True).sort_index()
    results['mcmc_fig'] = px.bar(x=mcmc_dist.index, y=mcmc_dist.values, title="MCMC Number Distribution")
    results['mcmc_pred'] = int(mcmc_dist.idxmax())
    results['trans_prob'] = trans_prob

    drift = np.diff(series).mean()
    diffusion = np.diff(series).var() / 2.0
    p = np.ones(max_num) / max_num
    x = np.arange(1, max_num + 1)
    dt, dx = 0.01, 1
    for _ in range(100):
        p_old = p.copy()
        for i in range(1, max_num-1):
            advective_term = -drift * (p_old[i+1] - p_old[i-1]) / (2 * dx)
            diffusive_term = diffusion * (p_old[i+1] - 2*p_old[i] + p_old[i-1]) / (dx**2)
            p[i] += dt * (advective_term + diffusive_term)
        p[0] = p[1]; p[-1] = p[-2]
        p = np.clip(p, 0, None)
        p /= p.sum()
    results['fokker_planck_fig'] = px.line(x=x, y=p, title="Fokker-Planck Probability Density")
    results['fokker_planck_pred'] = int(x[np.argmax(p)])

    return results

def _analyze_ml_models(series: np.ndarray, max_num: int) -> Dict[str, Any]:
    """Sub-module for Machine Learning analysis."""
    results = {}

    # HMM Model
    hmm_series = (series - 1).reshape(-1, 1)
    hmm = MultinomialHMM(n_components=5, n_iter=100, tol=1e-3, params='st', init_params='st')
    hmm.fit(hmm_series)
    last_state = hmm.predict(hmm_series)[-1]
    next_state = np.argmax(hmm.transmat_[last_state])
    # FIXED: Use the correct attribute name 'emission_prob_'
    hmm_pred = np.argmax(hmm.emission_prob_[next_state]) + 1
    results['hmm_pred'] = int(np.clip(hmm_pred, 1, max_num))
    return results


@st.cache_data
def analyze_stable_position_dynamics(_df: pd.DataFrame, position: str, max_num: int) -> Dict[str, Any]:
    """
    Performs a deep dive analysis on a single stable position.
    """
    try:
        results = {}
        series = _df[position].values
        
        stat_phys_results = _analyze_stat_physics(series, max_num)
        results.update(stat_phys_results)

        sarima_model = AutoARIMA(sp=1, suppress_warnings=True)
        sarima_model.fit(series)
        prediction_result = sarima_model.predict(fh=[1])
        sarima_pred = int(np.clip(np.round(prediction_result[0]), 1, max_num))
        results['sarima_pred'] = sarima_pred

        ml_results = _analyze_ml_models(series, max_num)
        results.update(ml_results)

        hist, _ = np.histogram(series, bins=max_num, range=(1, max_num+1), density=True)
        results['shannon_entropy'] = -np.sum(hist * np.log2(hist + 1e-12))

        preds = [results['mcmc_pred'], results['sarima_pred'], results['hmm_pred']]
        fuzzy_pred = int(np.clip(np.round(np.mean(preds)), 1, max_num))
        results['fuzzy_pred'] = fuzzy_pred

        return results
    except Exception as e:
        st.error(f"Error in stable position analysis for {position}: {e}")
        return {}


# --- 5. ADVANCED PREDICTIVE MODELS ---
@st.cache_resource
def train_torch_model(_df: pd.DataFrame, model_type: str = 'LSTM', seq_length: int = 3, epochs: int = 100) -> Tuple[Optional[nn.Module], Optional[MinMaxScaler], float]:
    """Trains a PyTorch sequence model (LSTM, GRU)."""
    try:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(_df)
        X, y = create_sequences(data_scaled, seq_length)
        X_torch, y_torch = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        class SequenceModel(nn.Module):
            def __init__(self, input_size=6, hidden_size=64):
                super().__init__()
                if model_type == 'LSTM':
                    self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)
                elif model_type == 'GRU':
                    self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)
                self.fc = nn.Linear(hidden_size, input_size)
            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :])

        model = SequenceModel().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X_torch), y_torch).item()

        return model, scaler, final_loss
    except Exception as e:
        st.error(f"Error training {model_type} model: {e}")
        return None, None, float('inf')

@st.cache_data
def predict_torch_model(_df: pd.DataFrame, _model_cache: Tuple, model_type: str, seq_length: int, max_nums: List[int]) -> Dict[str, Any]:
    """Generates a prediction using a trained PyTorch model."""
    try:
        model, scaler, best_loss = _model_cache
        if model is None or scaler is None:
            raise ValueError(f"{model_type} model training failed or cache is invalid.")

        last_seq = scaler.transform(_df.iloc[-seq_length:].values).reshape(1, seq_length, 6)
        last_seq_torch = torch.tensor(last_seq, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_scaled = model(last_seq_torch)

        prediction_raw = scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        prediction = np.round(prediction_raw).astype(int)

        clipped_preds = [np.clip(p, 1, max_nums[i]) for i, p in enumerate(prediction)]
        final_preds = []
        seen = set()
        for i, p_val in enumerate(clipped_preds):
            candidate = p_val
            while candidate in seen:
                candidate = np.random.randint(1, max_nums[i] + 1)
                if candidate not in seen:
                    break
            final_preds.append(candidate)
            seen.add(candidate)
        
        error_estimate = np.abs(prediction_raw - np.array(final_preds))

        return {
            'name': model_type,
            'prediction': sorted(final_preds),
            'error': error_estimate,
            'logic': f'Deep learning {model_type} sequence forecast.'
        }
    except Exception as e:
        st.warning(f"Prediction with {model_type} failed: {e}")
        return {'name': model_type, 'prediction': [0]*6, 'error': [0]*6, 'logic': 'Prediction failed.'}


@st.cache_data
def analyze_hilbert_embedding(_df: pd.DataFrame, max_nums: List[int]) -> Dict[str, Any]:
    """Predicts using a quantum-inspired geometric embedding in a complex space."""
    try:
        if len(_df) < 2: return {'name': 'Hilbert Space Embedding', 'prediction': [0]*6, 'error': [0]*6, 'logic': 'Insufficient data.'}
        
        def to_complex(n, pos_idx):
            return np.exp(1j * 2 * np.pi * n / (max_nums[pos_idx] + 1))

        complex_df = pd.DataFrame({col: _df[col].apply(lambda x: to_complex(x, i)) for i, col in enumerate(_df.columns)})
        mean_vector = complex_df.mean(axis=1)
        
        phase_velocity = np.angle(mean_vector.iloc[-1] / mean_vector.iloc[-2])
        amp_velocity = np.abs(mean_vector.iloc[-1]) - np.abs(mean_vector.iloc[-2])
        
        predicted_vector = (np.abs(mean_vector.iloc[-1]) + amp_velocity) * np.exp(1j * (np.angle(mean_vector.iloc[-1]) + phase_velocity))
        
        selected, seen = [], set()
        current_sum_vec = 0
        for pos_idx in range(6):
            potential_nums = np.arange(1, max_nums[pos_idx] + 1)
            potential_vectors = (current_sum_vec + to_complex(potential_nums, pos_idx)) / (len(selected) + 1)
            distances = np.abs(potential_vectors - predicted_vector)
            
            for num_idx in np.argsort(distances):
                best_num = potential_nums[num_idx]
                if best_num not in seen:
                    selected.append(best_num)
                    seen.add(best_num)
                    current_sum_vec += to_complex(best_num, pos_idx)
                    break
        
        return {
            'name': 'Hilbert Space Embedding',
            'prediction': sorted(selected),
            'error': np.full(6, np.min(distances) * 10),
            'logic': 'Predicts next draw\'s geometric center in a complex Hilbert space.'
        }
    except Exception as e:
        st.warning(f"Hilbert embedding failed: {e}")
        return {'name': 'Hilbert Space Embedding', 'prediction': [0]*6, 'error': [0]*6, 'logic': 'Failed due to error.'}


# --- 6. Backtesting & Meta-Analysis ---
@st.cache_data
def run_full_backtest_suite(_df: pd.DataFrame, max_nums: List[int], stable_positions: List[str]) -> List[Dict[str, Any]]:
    """
    Runs a walk-forward validation for all predictive models to derive a performance score.
    """
    try:
        scored_predictions = []
        split_point = max(50, int(len(_df) * 0.8))
        if len(_df) - split_point < 10:
            st.warning("Insufficient validation data for full backtest. Results may be less reliable.")
            split_point = len(_df) - 10

        train_df, val_df = _df.iloc[:split_point], _df.iloc[split_point:]
        
        lstm_cache = train_torch_model(train_df, 'LSTM')
        gru_cache = train_torch_model(train_df, 'GRU')

        model_funcs = {
            'LSTM': lambda d: predict_torch_model(d, lstm_cache, 'LSTM', 3, max_nums),
            'GRU': lambda d: predict_torch_model(d, gru_cache, 'GRU', 3, max_nums),
            'Hilbert Embedding': lambda d: analyze_hilbert_embedding(d, max_nums),
        }

        for pos in stable_positions:
            pos_idx = int(pos.split('_')[1]) - 1
            stable_results = analyze_stable_position_dynamics(train_df, pos, max_nums[pos_idx])
            
            def create_stable_pred_func(res, p_idx):
                def stable_pred_func(d):
                    base_pred = d.iloc[-1].values.copy()
                    pred_val = res.get('fuzzy_pred', d.iloc[-1, p_idx])
                    base_pred[p_idx] = pred_val
                    
                    final_preds, seen = [], set()
                    for val in base_pred:
                        candidate = val
                        while candidate in seen:
                            candidate = np.random.randint(1, max_nums[len(final_preds)] + 1)
                        final_preds.append(candidate)
                        seen.add(candidate)
                    
                    return {'name': f'Stable_{pos}', 'prediction': sorted(final_preds), 'error': [1.0]*6, 'logic': f'Ensemble pred for {pos}'}
                return stable_pred_func
            
            model_funcs[f'Stable_{pos}'] = create_stable_pred_func(stable_results, pos_idx)


        progress_bar = st.progress(0, text="Backtesting models...")
        total_steps = len(val_df) * len(model_funcs)
        current_step = 0

        for name, func in model_funcs.items():
            y_preds, y_trues = [], []
            for i in range(len(val_df)):
                historical_df = _df.iloc[:split_point + i]
                pred_obj = func(historical_df)
                if not pred_obj or sum(pred_obj['prediction']) == 0: continue
                
                y_preds.append(pred_obj['prediction'])
                y_trues.append(val_df.iloc[i].values.tolist())
                current_step += 1
                progress_bar.progress(min(1.0, current_step / total_steps), text=f"Backtesting {name} on draw {i+1}")

            if not y_preds:
                likelihood, metrics = 0, {'Avg Hits': 'N/A'}
            else:
                hits = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues, y_preds))
                rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
                avg_hits = hits / len(y_trues)
                likelihood = 0.7 * min(100, (avg_hits / 1.5) * 100) + 0.3 * max(0, 100 - rmse * 4)
                metrics = {'Avg Hits': f"{avg_hits:.2f}", 'RMSE': f"{rmse:.2f}"}

            final_pred_obj = func(_df)
            final_pred_obj['likelihood'] = likelihood
            final_pred_obj['metrics'] = metrics
            scored_predictions.append(final_pred_obj)

        progress_bar.empty()
        return sorted(scored_predictions, key=lambda x: x.get('likelihood', 0), reverse=True)
    except Exception as e:
        st.error(f"Error in backtesting suite: {e}")
        return []

# ====================================================================================================
# Main Application UI & Logic
# ====================================================================================================

st.title("⚛️ LottoSphere v16.0.15: Quantum Chronodynamics Engine")
st.markdown("A scientific instrument for exploratory analysis of high-dimensional, chaotic systems. Models each number position as an evolving system using advanced mathematical, AI, and statistical physics techniques.")

st.sidebar.header("Configuration")
seq_length = st.sidebar.slider("Sequence Length (for DL Models)", min_value=3, max_value=8, value=4, help="How many past draws to use for predicting the next one.")
max_nums = [st.sidebar.number_input(f"Max Number Pos_{i+1}", min_value=10, max_value=150, value=49 + (i * 2), key=f"max_num_pos_{i+1}") for i in range(6)]
epochs = st.sidebar.slider("Training Epochs", min_value=50, max_value=200, value=100, help="Number of training cycles for deep learning models.")
uploaded_file = st.sidebar.file_uploader("Upload Number History (CSV)", type=["csv"], help="CSV file with one draw per row, numbers in columns. Most recent draw should be the last row.")

if uploaded_file:
    df_master = load_data(uploaded_file, max_nums)
    
    for warning_msg in st.session_state.data_warnings:
        st.warning(warning_msg)

    if not df_master.empty and df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        
        with st.spinner("Analyzing system stability..."):
            stable_positions = [pos for pos in df_master.columns if analyze_temporal_behavior(df_master, pos).get('is_stable', False)]
        if stable_positions:
            st.sidebar.info(f"Stable positions detected: {', '.join(stable_positions)}. Specialized models will be available.")
        
        tab1, tab2 = st.tabs(["🔮 Predictive Analytics", "🔬 System Dynamics Explorer"])

        with tab1:
            st.header("Engage Grand Unified Predictive Ensemble")
            with st.expander("Explanation of Predictions & Methodology"):
                st.markdown("""
                This panel runs a suite of predictive models and ranks them based on historical performance.
                - **Backtesting**: Each model is tested on the last 20% of your data to simulate real-world performance.
                - **Likelihood Score**: A custom metric combining **Average Hits** (how many numbers it correctly guesses per draw) and **RMSE** (how close its numbers are to the actual winning numbers). A higher score indicates a more reliable model based on your data.
                - **Model Types**:
                    - **LSTM/GRU**: Advanced neural networks for time-series forecasting.
                    - **Hilbert Embedding**: A quantum-inspired geometric model.
                    - **Stable Models**: For positions identified as 'stable' (predictable), a specialized ensemble prediction is generated.
                - **Actionability**: Focus on models with a Likelihood Score > 50%. Consider combining numbers from the top 2-3 models for a diversified strategy.
                """)

            if st.button("🚀 RUN ALL PREDICTIVE MODELS", type="primary", use_container_width=True):
                with st.spinner("Backtesting all models... This may take several minutes."):
                    scored_predictions = run_full_backtest_suite(df_master, max_nums, stable_positions)
                
                st.header("✨ Final Synthesis & Strategic Portfolio")
                if scored_predictions:
                    st.subheader("Ranked Predictions by Historical Performance")
                    for p in scored_predictions:
                        with st.container(border=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"#### {p['name']}")
                                pred_str = ' | '.join([f"{n}" for n in p['prediction']])
                                st.markdown(f"**Candidate Set:** `{pred_str}`")
                                st.caption(f"Logic: {p['logic']}")
                            with col2:
                                st.metric("Likelihood Score", f"{p.get('likelihood', 0):.1f}%", help=f"Backtest Metrics: {p.get('metrics', {})}")

        with tab2:
            st.header("System Dynamics Explorer")
            st.markdown("Explore the intrinsic, time-dependent behavior of each number position.")
            position = st.selectbox("Select Position to Analyze", options=df_master.columns, index=0)

            with st.expander("What am I looking at?"):
                st.markdown("""
                This tool dissects the time series of a single number position.
                - **Recurrence Plot**: Visualizes when the system returns to a previous state. Dense patterns suggest predictability.
                - **Power Spectral Density**: Shows dominant cycles. A high peak might indicate a recurring pattern over a certain number of draws.
                - **Spectrogram**: Shows how these cycles change over time.
                - **Lyapunov Exponent**: A key metric from chaos theory.
                    - **Positive (> 0.05)**: The system is **chaotic** and unpredictable long-term.
                    - **Near-Zero or Negative**: The system is **stable** or periodic, making it more predictable.
                - **Stable Position Analysis**: If a position is stable, additional models (MCMC, HMM, SARIMA) are run to provide targeted predictions.
                """)

            if st.button(f"Analyze Dynamics for {position}", use_container_width=True):
                with st.spinner(f"Calculating dynamics for {position}..."):
                    dynamic_results = analyze_temporal_behavior(df_master, position=position)
                
                if dynamic_results:
                    st.subheader(f"Chaotic & Cyclical Analysis ({position})")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if dynamic_results['lyapunov'] > 0.05:
                            st.warning(f"**Lyapunov:** `{dynamic_results['lyapunov']:.4f}` (Chaotic)", icon="⚠️")
                        else:
                            st.success(f"**Lyapunov:** `{dynamic_results['lyapunov']:.4f}` (Stable)", icon="✅")
                    with col2:
                        if dynamic_results.get('periodicity'):
                            st.info(f"**Dominant Period:** `{dynamic_results['periodicity']}` draws", icon="🔄")
                    
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        st.plotly_chart(dynamic_results['recurrence_fig'], use_container_width=True)
                        if 'acf_fig' in dynamic_results:
                            st.plotly_chart(dynamic_results['acf_fig'], use_container_width=True)
                    with fig_col2:
                        st.plotly_chart(dynamic_results['fourier_fig'], use_container_width=True)
                        st.plotly_chart(dynamic_results['spectrogram_fig'], use_container_width=True)
                
                if position in stable_positions:
                    st.subheader(f"Advanced Stable Position Analysis ({position})")
                    with st.spinner(f"Running advanced models for stable position {position}..."):
                        stable_results = analyze_stable_position_dynamics(df_master, position, max_nums[df_master.columns.get_loc(position)])
                    
                    if stable_results:
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        pred_col1.metric("MCMC Prediction", stable_results.get('mcmc_pred', 'N/A'))
                        pred_col2.metric("HMM Prediction", stable_results.get('hmm_pred', 'N/A'))
                        pred_col3.metric("SARIMA Prediction", stable_results.get('sarima_pred', 'N/A'))

                        with st.expander("View Detailed Plots for Stable Analysis"):
                            st.plotly_chart(stable_results['mcmc_fig'], use_container_width=True)
                            st.plotly_chart(stable_results['fokker_planck_fig'], use_container_width=True)
else:
    st.info("Please upload a CSV file to begin analysis.")
