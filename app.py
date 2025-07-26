# ======================================================================================================
# LottoSphere v17.0.2: The Quantum Chronodynamics Engine (Final Corrected)
#
# VERSION: 17.0.2 (Final Corrected)
#
# DESCRIPTION:
# This is the final, stable, and feature-complete version of the application, incorporating
# probabilistic forecasting and all accumulated bug fixes. Models each of six sorted number
# positions as an independent yet interacting dynamical system, using deep learning, statistical
# physics, and time-series analysis. Enhanced with spectrogram-based time-frequency analysis
# and robust error handling for stable operation.
#
# CHANGELOG:
# - v17.0.2: Removed tensorflow dependencies, simplified to PyTorch, added probabilistic forecasting.
# - Fixed streamlit-rich conflict, removed pywt, added robust error handling.
# - Fixed HMM attribute error ('emissionprob_' to 'emissionprobs_') for hmmlearn==0.3.2.
# - Fixed typo 'analyze_teminal_behavior' to 'analyze_temporal_behavior' in main logic.
# - Fixed syntax error in MultinomialHMM ('params nonin' to 'params') in _analyze_ml_models.
# - Restored HMM attribute fix ('emissionprob_' to 'emissionprobs_') and improved HMM call logging.
# - Added cache-clearing mechanism to ensure updated code execution.
# - Fixed UnboundLocalError for 'call_id' in _analyze_ml_models by defining it outside try block.
# - Ensured predictions respect max_nums and temporal CSV order (last rows as recent draws).
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
import scipy.stats as stats
import os

# --- Suppress Warnings for a Cleaner UI ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Advanced Scientific & ML Libraries ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, log_loss
from scipy.signal import welch, spectrogram
from nolds import lyap_r
from statsmodels.tsa.stattools import acf
from prophet import Prophet
from hmmlearn.hmm import MultinomialHMM
from sktime.forecasting.arima import AutoARIMA

# --- Deep Learning (PyTorch) ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. APPLICATION CONFIGURATION & INITIALIZATION ---
st.set_page_config(
    page_title="LottoSphere v17.0.2: Quantum Chronodynamics",
    page_icon="⚛️",
    layout="wide",
)
if 'data_warnings' not in st.session_state:
    st.session_state.data_warnings = []
if 'hmm_calls' not in st.session_state:
    st.session_state.hmm_calls = {}
if 'cache_cleared' not in st.session_state:
    st.session_state.cache_cleared = False

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Clear Streamlit Cache ---
if not st.session_state.cache_cleared:
    st.cache_data.clear()
    st.session_state.cache_cleared = True
    st.session_state.data_warnings.append("Streamlit cache cleared to ensure updated code execution.")

# ====================================================================================================
# ALL FUNCTION DEFINITIONS
# ====================================================================================================

# --- 2. CORE UTILITIES & DATA HANDLING ---
@st.cache_data
def load_data(uploaded_file: io.BytesIO, max_nums: List[int]) -> pd.DataFrame:
    """Loads, validates, and preprocesses the lottery data from a CSV file."""
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
            st.session_state.data_warnings.append(f"Discarded {initial_rows - len(df)} rows with numbers outside the specified max range.")

        num_cols = min(6, df.shape[1])
        df = df.iloc[:, :num_cols]
        unique_counts = df.apply(lambda x: len(set(x)), axis=1)
        valid_rows_mask = (unique_counts == num_cols)
        if not valid_rows_mask.all():
            st.session_state.data_warnings.append(f"Discarded {len(df) - valid_rows_mask.sum()} rows with duplicate numbers within the draw.")
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

def get_best_guess_set(distributions: List[Dict[int, float]], max_nums: List[int]) -> List[int]:
    """Extracts a unique set of numbers from the modes of probability distributions."""
    best_guesses = []
    seen = set()
    for i, dist in enumerate(distributions):
        if not dist:  # Handle empty distribution
            available = set(range(1, max_nums[i] + 1)) - seen
            guess = np.random.choice(list(available)) if available else np.random.randint(1, max_nums[i] + 1)
        else:
            sorted_dist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
            for num, _ in sorted_dist:
                if num not in seen:
                    guess = num
                    break
            else:
                available = set(range(1, max_nums[i] + 1)) - seen
                guess = np.random.choice(list(available)) if available else np.random.randint(1, max_nums[i] + 1)
        best_guesses.append(guess)
        seen.add(guess)
    return sorted(best_guesses)

# --- 3. TIME-DEPENDENT BEHAVIOR ANALYSIS MODULE ---
@st.cache_data
def analyze_temporal_behavior(_df: pd.DataFrame, position: str = 'Pos_1') -> Dict[str, Any]:
    """Analyzes the chaotic and cyclical nature of a single positional time series."""
    try:
        results = {}
        series = _df[position].values
        if len(series) < 10:  # Prevent crashes on short series
            st.warning(f"Insufficient data for {position} analysis (<10 draws).")
            return results

        # Recurrence Plot
        recurrence_matrix = np.abs(np.subtract.outer(series, series))
        normalized_recurrence = recurrence_matrix / (recurrence_matrix.max() + 1e-10)
        results['recurrence_fig'] = px.imshow(normalized_recurrence, color_continuous_scale='viridis', title=f"Recurrence Plot ({position})")

        # Fourier Analysis
        freqs, psd = welch(series, nperseg=min(len(series), 256))
        psd_df = pd.DataFrame({'Frequency': freqs, 'Power': psd}).sort_values('Power', ascending=False)
        results['fourier_fig'] = px.line(psd_df, x='Frequency', y='Power', title=f"Power Spectral Density ({position})")

        # Spectrogram
        f, t, Sxx = spectrogram(series, fs=1.0, nperseg=min(128, len(series)//2), noverlap=int(min(128, len(series)//2)*0.9))
        results['spectrogram_fig'] = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx + 1e-10), x=t, y=f, colorscale='viridis'))
        results['spectrogram_fig'].update_layout(title=f'Spectrogram ({position})', xaxis_title='Time', yaxis_title='Frequency')

        # Lyapunov Exponent
        try:
            lyap_exp = lyap_r(series, emb_dim=max(2, len(series)//10), lag=1, min_tsep=len(series)//10)
            results['lyapunov'] = lyap_exp
            results['is_stable'] = lyap_exp <= 0.05
        except Exception:
            results['lyapunov'] = float('nan')
            results['is_stable'] = False
            st.warning(f"Lyapunov exponent calculation failed for {position}.")

        # Periodicity Analysis for Stable Positions
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
        st.error(f"Error in temporal behavior analysis for {position}: {e}")
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
    mcmc_samples = [np.random.choice(max_num, p=prob_vector) for _ in range(5000)]
    mcmc_dist = pd.Series(c + 1 for c in mcmc_samples).value_counts(normalize=True).sort_index()
    results['mcmc_dist'] = mcmc_dist.to_dict()
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
        p[0], p[-1] = p[1], p[-2]
        p = np.clip(p, 0, None)
        p /= p.sum()
    results['fokker_planck_dist'] = {int(num): prob for num, prob in zip(x, p)}
    return results

def _analyze_ml_models(series: np.ndarray, max_num: int, position: str) -> Dict[str, Any]:
    """Sub-module for Machine Learning analysis."""
    results = {}
    hmm_series = (series - 1).reshape(-1, 1)
    call_id = f"{position}_{len(st.session_state.hmm_calls.get(position, {})) + 1}"
    st.session_state.hmm_calls.setdefault(position, {})[call_id] = st.session_state.hmm_calls.get(position, {}).get(call_id, 0) + 1
    try:
        # Validate input series
        if not np.all((hmm_series >= 0) & (hmm_series < max_num)):
            raise ValueError(f"Invalid values in series for {position}: must be in range [0, {max_num-1}]")
        hmm = MultinomialHMM(n_components=5, n_iter=100, tol=1e-3, params='st', init_params='st')
        hmm.fit(hmm_series)
        last_state = hmm.predict(hmm_series)[-1]
        next_state = np.argmax(hmm.transmat_[last_state])
        # Use correct attribute name for hmmlearn==0.3.2
        emission_probs = getattr(hmm, 'emissionprobs_', None)
        if emission_probs is None:
            st.warning(f"HMM emission probabilities not available for {position} (call {call_id}).")
            results['hmm_dist'] = {i: 1/max_num for i in range(1, max_num + 1)}
        else:
            results['hmm_dist'] = {i+1: p for i, p in enumerate(emission_probs[next_state]) if 1 <= i+1 <= max_num}
    except Exception as e:
        st.warning(f"HMM model failed for {position} (call {call_id}): {e}")
        results['hmm_dist'] = {i: 1/max_num for i in range(1, max_num + 1)}
    return results

@st.cache_data
def analyze_stable_position_dynamics(_df: pd.DataFrame, position: str, max_num: int) -> Dict[str, Any]:
    """Performs a deep dive analysis on a single stable position."""
    try:
        results = {}
        series = _df[position].values
        stat_phys_results = _analyze_stat_physics(series, max_num)
        results.update(stat_phys_results)
        try:
            sarima_model = AutoARIMA(sp=1, suppress_warnings=True)
            sarima_model.fit(series)
            pred_point = sarima_model.predict(fh=[1])[0]
            conf_int = sarima_model.predict_interval(fh=[1], coverage=0.95)
            std_dev = max(1.0, (conf_int.iloc[0, 1] - conf_int.iloc[0, 0]) / 3.92)
            x_range = np.arange(1, max_num + 1)
            prob_mass = stats.norm.pdf(x_range, loc=pred_point, scale=std_dev)
            prob_mass /= prob_mass.sum()
            results['sarima_dist'] = {int(num): prob for num, prob in zip(x_range, prob_mass)}
        except Exception:
            st.warning(f"SARIMA model failed for {position}.")
            results['sarima_dist'] = {i: 1/max_num for i in range(1, max_num + 1)}
        ml_results = _analyze_ml_models(series, max_num, position)
        results.update(ml_results)
        all_dists = [results.get('mcmc_dist', {}), results.get('sarima_dist', {}), results.get('hmm_dist', {})]
        ensemble_dist = {i: 0.0 for i in range(1, max_num + 1)}
        for dist in all_dists:
            for num, prob in dist.items():
                if 1 <= num <= max_num:
                    ensemble_dist[num] += prob
        total_prob = sum(ensemble_dist.values()) or 1
        results['distributions'] = [{num: prob / total_prob for num, prob in ensemble_dist.items()}]
        return results
    except Exception as e:
        st.error(f"Error in stable position analysis for {position}: {e}")
        return {'distributions': [{i: 1/max_num for i in range(1, max_num + 1)}]}

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
                self.rnn = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=0.1) if model_type == 'LSTM' else nn.GRU(input_size, hidden_size, 2, batch_first=True, dropout=0.1)
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
    """Generates a probabilistic prediction using a trained PyTorch model."""
    try:
        model, scaler, best_loss = _model_cache
        if model is None:
            st.warning(f"{model_type} model training failed.")
            return {'name': model_type, 'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in max_nums], 'logic': f'{model_type} failed.'}
        last_seq = scaler.transform(_df.iloc[-seq_length:].values).reshape(1, seq_length, 6)
        last_seq_torch = torch.tensor(last_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(last_seq_torch)
        prediction_raw = scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        std_dev = max(1.5, np.sqrt(best_loss) * (max_nums[0] - 1))
        distributions = []
        for i in range(6):
            max_num = max_nums[i]
            x_range = np.arange(1, max_num + 1)
            prob_mass = stats.norm.pdf(x_range, loc=prediction_raw[i], scale=std_dev)
            prob_mass /= prob_mass.sum()
            distributions.append({int(num): prob for num, prob in zip(x_range, prob_mass)})
        return {'name': model_type, 'distributions': distributions, 'logic': f'Deep learning {model_type} sequence forecast.'}
    except Exception as e:
        st.warning(f"Prediction with {model_type} failed: {e}")
        return {'name': model_type, 'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in max_nums], 'logic': f'{model_type} failed.'}

# --- 6. Backtesting & Meta-Analysis ---
@st.cache_data
def run_full_backtest_suite(_df: pd.DataFrame, max_nums: List[int], stable_positions: List[str]) -> List[Dict[str, Any]]:
    """Runs a walk-forward validation for all models using Log Loss for scoring."""
    try:
        scored_predictions = []
        split_point = max(50, int(len(_df) * 0.8))
        if len(_df) - split_point < 10:
            st.warning("Insufficient validation data for full backtest.")
            split_point = len(_df) - 10
        train_df, val_df = _df.iloc[:split_point], _df.iloc[split_point:]
        lstm_cache = train_torch_model(train_df, 'LSTM')
        gru_cache = train_torch_model(train_df, 'GRU')
        model_funcs = {
            'LSTM': lambda d: predict_torch_model(d, lstm_cache, 'LSTM', 3, max_nums),
            'GRU': lambda d: predict_torch_model(d, gru_cache, 'GRU', 3, max_nums),
        }
        if stable_positions:  # Only add stable position models if available
            for pos in stable_positions:
                model_funcs[f'Stable_{pos}'] = lambda d, p=pos: analyze_stable_position_dynamics(d, p, max_nums[int(p.split("_")[1])-1])
        progress_bar = st.progress(0, text="Backtesting models...")
        total_steps = len(val_df) * len(model_funcs)
        current_step = 0
        for name, func in model_funcs.items():
            total_log_loss, draw_count = 0, 0
            for i in range(len(val_df)):
                historical_df = _df.iloc[:split_point + i]
                pred_obj = func(historical_df)
                if not pred_obj or not pred_obj.get('distributions'):
                    continue
                y_true = val_df.iloc[i].values
                draw_log_loss = 0
                for pos_idx, dist in enumerate(pred_obj['distributions']):
                    true_num = y_true[pos_idx]
                    prob_of_true = dist.get(true_num, 1e-9)
                    draw_log_loss -= np.log(prob_of_true)
                total_log_loss += draw_log_loss
                draw_count += 1
                current_step += 1
                progress_bar.progress(min(1.0, current_step / total_steps), text=f"Backtesting {name}...")
            avg_log_loss = (total_log_loss / draw_count) if draw_count > 0 else 99
            likelihood = max(0, 100 - avg_log_loss * 15)
            final_pred_obj = func(_df)
            final_pred_obj['likelihood'] = likelihood
            final_pred_obj['metrics'] = {'Avg Log Loss': f"{avg_log_loss:.3f}"}
            final_pred_obj['prediction'] = get_best_guess_set(final_pred_obj['distributions'], max_nums)
            scored_predictions.append(final_pred_obj)
        progress_bar.empty()
        st.session_state.data_warnings.append(f"Total HMM calls during backtest: {sum([sum(calls.values()) for calls in st.session_state.hmm_calls.values()])}")
        return sorted(scored_predictions, key=lambda x: x.get('likelihood', 0), reverse=True)
    except Exception as e:
        st.error(f"Error in backtesting suite: {e}")
        return []

# ====================================================================================================
# Main Application UI & Logic
# ====================================================================================================
st.title("⚛️ LottoSphere v17.0.2: Quantum Chronodynamics Engine")
st.markdown("A scientific instrument for exploratory analysis, now upgraded with **probabilistic forecasting**.")

st.sidebar.header("Configuration")
max_nums = [st.sidebar.number_input(f"Max Number Pos_{i+1}", min_value=10, max_value=150, value=49 + (i * 2), key=f"max_num_pos_{i+1}") for i in range(6)]
uploaded_file = st.sidebar.file_uploader("Upload Number History (CSV)", type=["csv"], help="CSV file with one draw per row, numbers in columns. Most recent draw should be the last row.")

if uploaded_file:
    df_master = load_data(uploaded_file, max_nums)
    for warning_msg in st.session_state.data_warnings:
        st.warning(warning_msg)
    if not df_master.empty and df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        with st.spinner("Analyzing system stability..."):
            stable_positions = []
            for pos in df_master.columns:
                result = analyze_temporal_behavior(df_master, pos)
                if result.get('is_stable', False):
                    stable_positions.append(pos)
                st.session_state.data_warnings.append(f"{pos} stability: Lyapunov={result.get('lyapunov', 'N/A')}, Stable={result.get('is_stable', False)}")
        if stable_positions:
            st.sidebar.info(f"Stable positions detected: {', '.join(stable_positions)}.")
        else:
            st.sidebar.info("No stable positions detected.")
        tab1, tab2 = st.tabs(["🔮 Probabilistic Forecasting", "🔬 System Dynamics Explorer"])
        with tab1:
            st.header("Engage Grand Unified Predictive Ensemble")
            with st.expander("Explanation of Probabilistic Forecasts"):
                st.markdown("""
                ### Overview
                The Probabilistic Forecasting tab generates probability distributions for the next lottery draw using LSTM, GRU, and stable position models (MCMC, SARIMA, HMM). Predictions are ranked by historical performance (log loss) on a validation set.

                ### Models
                - **LSTM/GRU**: Deep learning models capturing sequential patterns.
                - **Stable Position Models**: For stable positions (Lyapunov exponent ≤ 0.05), combine MCMC (Markov Chain Monte Carlo), SARIMA (time-series forecasting), and HMM (Hidden Markov Model).
                - **Backtesting**: Uses walk-forward validation to compute average log loss, converted to a likelihood score (100 - 15 × log loss).

                ### Actionability
                - Select high-likelihood predictions (>70%) for number sets.
                - Review probability distributions for each position to assess confidence.
                - Cross-validate with System Dynamics Explorer for stable positions.
                """)
            if st.button("🚀 RUN ALL PREDICTIVE MODELS", type="primary", use_container_width=True):
                with st.spinner("Backtesting all models... This is a deep computation and may take several minutes."):
                    scored_predictions = run_full_backtest_suite(df_master, max_nums, stable_positions)
                st.header("✨ Final Synthesis & Strategic Portfolio")
                if scored_predictions:
                    st.subheader("Ranked Probabilistic Forecasts by Historical Performance")
                    for p in scored_predictions:
                        with st.container(border=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"#### {p.get('name', 'Unknown Model')}")
                                pred_str = ' | '.join([f"{n}" for n in p['prediction']])
                                st.markdown(f"**Most Likely Set:** `{pred_str}`")
                            with col2:
                                st.metric("Likelihood Score", f"{p.get('likelihood', 0):.1f}%", help=f"Backtest Metrics: {p.get('metrics', {})}")
                            with st.expander("View Full Probability Distributions"):
                                chart_cols = st.columns(6)
                                for i, dist in enumerate(p['distributions']):
                                    with chart_cols[i]:
                                        st.markdown(f"**Pos_{i+1}**")
                                        df_dist = pd.DataFrame(list(dist.items()), columns=['Number', 'Probability']).sort_values('Number')
                                        fig = px.bar(df_dist, x='Number', y='Probability', height=200)
                                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None, xaxis_title=None)
                                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with tab2:
            st.header("System Dynamics Explorer")
            with st.expander("Explanation of System Dynamics"):
                st.markdown("""
                ### Overview
                Analyzes the temporal behavior of a selected position using chaos theory and time-series analysis.

                ### Outputs
                - **Recurrence Plot**: Visualizes state recurrences.
                - **Power Spectral Density (Fourier)**: Identifies dominant cycles.
                - **Spectrogram**: Maps time-varying frequency content.
                - **Lyapunov Exponent**: Quantifies chaos (positive) or stability (≤0.05).
                - **Periodicity Analysis**: Detects cycles for stable positions.

                ### Actionability
                - Stable positions (Lyapunov ≤0.05) suggest predictable patterns; use their predictions in Tab 1.
                - Cross-reference periodicity with forecast distributions.
                """)
            position = st.selectbox("Select Position to Analyze", options=df_master.columns, index=0)
            if st.button(f"Analyze Dynamics for {position}", use_container_width=True):
                with st.spinner(f"Analyzing dynamics for {position}..."):
                    dynamic_results = analyze_temporal_behavior(df_master, position=position)
                if dynamic_results:
                    st.subheader(f"Chaotic & Cyclical Analysis ({position})")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(dynamic_results.get('recurrence_fig'), use_container_width=True)
                    with col2:
                        st.plotly_chart(dynamic_results.get('fourier_fig'), use_container_width=True)
                    st.plotly_chart(dynamic_results.get('spectrogram_fig'), use_container_width=True)
                    if not np.isnan(dynamic_results.get('lyapunov', float('nan'))):
                        if dynamic_results['lyapunov'] > 0.05:
                            st.warning(f"**Lyapunov Exponent:** {dynamic_results['lyapunov']:.4f}. System is chaotic.", icon="⚠️")
                        else:
                            st.success(f"**Lyapunov Exponent:** {dynamic_results['lyapunov']:.4f}. System is stable.", icon="✅")
                            if dynamic_results.get('periodicity_description'):
                                st.info(f"**Periodicity Analysis:** {dynamic_results['periodicity_description']}", icon="🔄")
                                st.plotly_chart(dynamic_results.get('acf_fig'), use_container_width=True)
                    else:
                        st.warning("Lyapunov exponent calculation failed.")
else:
    st.info("Please upload a CSV file to begin analysis.")
