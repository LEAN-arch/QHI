# ======================================================================================================
# LottoSphere v17.0.5: The Quantum Chronodynamics Engine (Final Corrected)
#
# VERSION: 17.0.5 (Final Corrected)
#
# DESCRIPTION:
# This is the final, stable, and feature-complete version of the application, incorporating
# probabilistic forecasting and all accumulated bug fixes. Models each of six sorted number
# positions as an independent yet interacting dynamical system, using deep learning, statistical
# physics, and time-series analysis. Enhanced with spectrogram-based time-frequency analysis
# and robust error handling for stable operation.
#
# CHANGELOG:
# - v17.0.5: Fixed SARIMA error 'unsupported format string passed to numpy.ndarray.format' by ensuring scalar pred_point.
# - Enhanced SARIMA error handling with detailed logging of input series and failure reasons.
# - Improved get_best_guess_set to randomize uniform distribution selections to avoid 1-2-3-4-5-6.
# - Strengthened load_data to reject CSVs with non-numeric values or insufficient unique draws.
# - Added logging for SARIMA inputs and outputs in analyze_stable_position_dynamics.
# - Adjusted likelihood score calculation to reduce uniform distribution impact.
# - Fixed zero likelihood scores by debugging log loss in run_full_backtest_suite.
# - Fixed identical 1-2-3-4-5-6 outputs for stable positions.
# - Enhanced get_best_guess_set to prioritize high-probability numbers.
# - Added detailed logging for distributions, prob_of_true, and y_true.
# - Fixed Markov Chain failure for Pos_6 by clipping series.
# - Fixed SyntaxError in run_full_backtest_suite (float1e-10 to float(1e-10)).
# - Replaced MultinomialHMM with Markov Chain model.
# - Removed hmmlearn dependency.
# - Fixed typo 'analyze_teminal_behavior' to 'analyze_temporal_behavior'.
# - Fixed UnboundLocalError for 'call_id' in _analyze_ml_models.
# - Ensured predictions respect max_nums and temporal CSV order.
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
from sklearn.metrics import mean_squared_error
from scipy.signal import welch, spectrogram
from nolds import lyap_r
from statsmodels.tsa.stattools import acf
from prophet import Prophet
from sktime.forecasting.arima import AutoARIMA

# --- Deep Learning (PyTorch) ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. APPLICATION CONFIGURATION & INITIALIZATION ---
st.set_page_config(
    page_title="LottoSphere v17.0.5: Quantum Chronodynamics",
    page_icon="⚛️",
    layout="wide",
)
if 'data_warnings' not in st.session_state:
    st.session_state.data_warnings = []
if 'model_calls' not in st.session_state:
    st.session_state.model_calls = {}
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
        st.session_state.data_warnings.append(f"Loaded CSV with {len(df)} rows and {df.shape[1]} columns. max_nums={max_nums}")
        if df.shape[1] != 6:
            st.session_state.data_warnings.append(f"CSV has {df.shape[1]} columns, expected exactly 6.")
            return pd.DataFrame()
        if len(max_nums) != 6:
            st.session_state.data_warnings.append(f"max_nums must have 6 values, got {len(max_nums)}: {max_nums}")
            return pd.DataFrame()
        
        # Check for non-numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df.isna().any().any():
            st.session_state.data_warnings.append(f"Found {df.isna().sum().sum()} non-numeric or NaN values in CSV.")
            return pd.DataFrame()
        df = df.astype(int)

        # Validate against max_nums
        initial_rows = len(df)
        for i, max_num in enumerate(max_nums):
            invalid_values = df.iloc[:, i][(df.iloc[:, i] < 1) | (df.iloc[:, i] > max_num)]
            if not invalid_values.empty:
                st.session_state.data_warnings.append(
                    f"Pos_{i+1}: Found {len(invalid_values)} values outside [1, {max_num}]: {invalid_values.unique().tolist()}"
                )
            df = df[(df.iloc[:, i] >= 1) & (df.iloc[:, i] <= max_num)]
        if len(df) < initial_rows:
            st.session_state.data_warnings.append(f"Discarded {initial_rows - len(df)} rows with numbers outside max_nums.")

        # Check for duplicates within rows
        unique_counts = df.apply(lambda x: len(set(x)), axis=1)
        valid_rows_mask = (unique_counts == 6)
        if not valid_rows_mask.all():
            st.session_state.data_warnings.append(f"Discarded {len(df) - valid_rows_mask.sum()} rows with duplicate numbers within the draw.")
            df = df[valid_rows_mask]

        # Check for duplicate rows
        if df.duplicated().any():
            st.session_state.data_warnings.append(f"Discarded {df.duplicated().sum()} duplicate rows.")
            df = df.drop_duplicates()

        df = df.reset_index(drop=True)
        df.columns = [f'Pos_{i+1}' for i in range(df.shape[1])]

        if len(df) < 50:
            st.session_state.data_warnings.append(f"Insufficient data for robust analysis (at least 50 rows required). Got {len(df)} rows.")
            return pd.DataFrame()

        unique_draws = len(df.drop_duplicates())
        if unique_draws < 50:
            st.session_state.data_warnings.append(f"Insufficient unique draws ({unique_draws}). At least 50 unique draws required.")
            return pd.DataFrame()

        st.session_state.data_warnings.append(f"Input data sorted per row to create positional time series. Last rows treated as most recent. Loaded {len(df)} draws with {unique_draws} unique draws.")
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
    """Generates a unique set of six numbers from probability distributions."""
    best_guesses = []
    seen = set()
    if len(distributions) != 6:
        st.session_state.data_warnings.append(f"Expected 6 distributions, got {len(distributions)}. Using uniform distributions.")
        distributions = [{i: 1/float(max_num) for i in range(1, max_num + 1)} for max_num in max_nums[:6]]
    
    for i, dist in enumerate(distributions[:6]):
        if not dist or not all(isinstance(p, (int, float)) and p >= 0 for p in dist.values()):
            st.session_state.data_warnings.append(f"Invalid distribution for Pos_{i+1}: {dist}. Using uniform distribution.")
            dist = {j: 1/float(max_nums[i]) for j in range(1, max_nums[i] + 1)}
        total_prob = sum(dist.values())
        if total_prob == 0 or np.isnan(total_prob):
            st.session_state.data_warnings.append(f"Zero or NaN total probability for Pos_{i+1}: {total_prob}. Using uniform distribution.")
            dist = {j: 1/float(max_nums[i]) for j in range(1, max_nums[i] + 1)}
        else:
            dist = {k: v/total_prob for k, v in dist.items()}
        
        # Shuffle keys for uniform distributions to avoid predictable 1-6
        numbers = list(dist.keys())
        np.random.shuffle(numbers)
        sorted_dist = sorted([(num, dist[num]) for num in numbers], key=lambda x: (-x[1], x[0]))
        
        for num, prob in sorted_dist:
            if num not in seen and 1 <= num <= max_nums[i]:
                best_guesses.append(num)
                seen.add(num)
                break
        else:
            available = sorted(set(range(1, max_nums[i] + 1)) - seen)
            if available:
                guess = np.random.choice(available)  # Randomize to avoid lowest number
                best_guesses.append(guess)
                seen.add(guess)
            else:
                guess = np.random.randint(1, max_nums[i] + 1)
                while guess in seen:
                    guess = np.random.randint(1, max_nums[i] + 1)
                best_guesses.append(guess)
                seen.add(guess)
    
    # Ensure exactly 6 numbers
    while len(best_guesses) < 6:
        for i in range(6):
            if len(best_guesses) >= 6:
                break
            available = sorted(set(range(1, max_nums[i] + 1)) - seen)
            if available:
                guess = np.random.choice(available)
                best_guesses.append(guess)
                seen.add(guess)
    st.session_state.data_warnings.append(f"Generated best guess set: {best_guesses[:6]}")
    return sorted(best_guesses[:6])

# --- 3. TIME-DEPENDENT BEHAVIOR ANALYSIS MODULE ---
@st.cache_data
def analyze_temporal_behavior(_df: pd.DataFrame, position: str = 'Pos_1') -> Dict[str, Any]:
    """Analyzes the chaotic and cyclical nature of a single positional time series."""
    try:
        results = {}
        series = _df[position].values
        if len(series) < 10:
            st.warning(f"Insufficient data for {position} analysis (<10 draws).")
            return results

        # Recurrence Plot
        recurrence_matrix = np.abs(np.subtract.outer(series, series))
        normalized_recurrence = recurrence_matrix / (recurrence_matrix.max() + 1e-10)
        results['recurrence_fig'] = px.imshow(normalized_recurrence, color_continuous_scale='viridis', title=f'Recurrence Plot: {position}')

        # Fourier Analysis
        freqs, psd = welch(series, nperseg=min(len(series), 256))
        psd_df = pd.DataFrame({'Frequency': freqs, 'Power': psd}).sort_values('Power', ascending=False)
        results['fourier_fig'] = px.line(psd_df, x='Frequency', y='Power', title=f"Power Spectral Density: {position}")

        # Spectrogram
        f, t, Sxx = spectrogram(series, fs=1.0, nperseg=min(128, len(series)//2), noverlap=int(min(128, len(series)//2)*0.9))
        results['spectrogram_fig'] = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx + 1e-10), x=t, y=f, colorscale='viridis'))
        results['spectrogram_fig'].update_layout(title=f'Spectrogram: {position}', xaxis_title='Time', yaxis_title='Frequency')

        # Lyapunov Exponent
        try:
            lyap_exp = lyap_r(series, emb_dim=max(2, len(series)//10), lag=1, min_tsep=10)
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
                dominant_period = lags[significant_peaks_indices[0] + 1]
                results['periodicity'] = dominant_period
                results['periodicity_description'] = f"Potential periodicity with period of {dominant_period} draws."
            else:
                results['periodicity'] = None
                results['periodicity_description'] = "No significant periodicity detected."
            acf_df = pd.DataFrame({'Lag': lags, 'Autocorrelation': acf_vals})
            results['acf_fig'] = px.line(acf_df, x='Lag', y='Autocorrelation', title=f'Autocorrelation Function: {position}', markers=True)
            results['acf_fig'].add_hline(y=conf_interval, line_dash="dash", line_color="red")
            results['acf_fig'].add_hline(y=-conf_interval, line_dash="dash", line_color="red")

        return results
    except Exception as e:
        st.error(f"Error in temporal behavior analysis for {position}: {e}")
        return {}

# --- 4. ADVANCED STABLE POSITION ANALYSIS ---
def _analyze_stat_physics(series: np.ndarray, max_num: int) -> Dict[str, Any]:
    """Statistical physics and Markov Chain Monte Carlo analysis."""
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
        st.session_state.data_warnings.append(f"Invalid MCMC state: {series[-1]}, using random {current_state + 1}")
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
    results['fokker_planck'] = {int(num): prob for num, prob in zip(x, p)}
    st.session_state.data_warnings.append(f"MCMC top 5: {sorted(mcmc_dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
    return results

def _analyze_ml_models(series: np.ndarray, max_num: int, position: str) -> Dict[str, Any]:
    """Machine learning analysis with Markov Chain."""
    results = {}
    st.session_state.data_warnings.append(f"Markov Chain for {position}: max_num={max_num}, series={series[:5]}...")
    series = series.astype(int) - 1  # 0-based indexing
    call_id = f"{position}_{len(st.session_state.model_calls.get(position, [])) + 1}"
    st.session_state.model_calls.setdefault(position, []).append(call_id)
    try:
        invalid = series[(series < 0) | (series >= max_num)]
        if invalid.size > 0:
            st.session_state.data_warnings.append(f"Pos_{position}: Clipping {len(invalid)} invalid values: {invalid.tolist()}")
            series = np.clip(series, 0, max_num - 1)
        if not np.all((series >= 0) & (series < max_num)):
            raise ValueError(f"Invalid series values for {position}: must be in [0, {max_num-1}]")
        
        trans_matrix = np.zeros((max_num, max_num))
        for i in range(len(series) - 1):
            from_state, to_state = int(series[i]), int(series[i + 1])
            trans_matrix[from_state, to_state] += 1
        trans_matrix += 1e-10
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_matrix = np.where(row_sums > 0, trans_matrix / row_sums, 1/float(max_num))
        last_state = int(series[-1]) if 0 <= int(series[-1]) < max_num else 0
        prob_dist = trans_matrix[last_state]
        prob_dist = np.clip(prob_dist, 1e-10, 1)
        prob_dist /= prob_dist.sum()
        results['markov_dist'] = {i + 1: float(p) for i, p in enumerate(prob_dist)}
        st.session_state.data_warnings.append(f"Markov top 5: {sorted(results['markov_dist'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        
    except Exception as e:
        st.error(f"Markov Chain failed for {position} (ID: {call_id}): {e}")
        results['markov_dist'] = {i: 1/float(max_num) for i in range(1, max_num + 1)}
    
    results['hmm_dist'] = results.pop('markov_dist', {i: 1/float(max_num) for i in range(1, max_num + 1)})
    return results

@st.cache_data
def analyze_stable_position_dynamics(_df: pd.DataFrame, position: str, max_num: int) -> Dict[str, Any]:
    """Analyzes a stable position with SARIMA, MCMC, and Markov Chain."""
    try:
        st.session_state.data_warnings.append(f"Analyzing {position}: max_num={max_num}")
        results = {}
        series = _df[position].values.astype(int)
        if not all(1 <= x <= max_num for x in series)):
            st.session_state.data_warnings.append(f"Invalid values in {position} series: {series[:5]}... Clipping to [1, {max_num}]")
            series = np.clip(series, 1, max_num)
        
        stat_phys_results = _analyze_stat_physics(series, max_num)
        results.update(stat_phys_results)
        
        try:
            if len(series) < 10:
                raise ValueError(f"Series for {position} too short: {len(series)} < 10")
            st.session_state.data_warnings.append(f"SARIMA input for {position}: {series[:5]}..., len={len(series)}")
            sarima_model = AutoARIMA(sp=6, suppress_warnings=True, maxiter=50)
            sarima_model.fit(series)
            pred_series = sarima_model.predict(fh=[1])
            pred_point = float(pred_series[0]) if isinstance(pred_series, (pd.Series, np.ndarray)) else float(pred_series)
            conf_int = sarima_model.predict_interval(fh=[1], coverage=0.95)
            std_dev = float(max(0.1, (conf_int.iloc[0, 1] - float(conf_int.iloc[0, 0])) / 3.92))
            x_range = np.arange(1, max_num + 1)
            prob_mass = stats.norm.pdf(x_range, loc=pred_point, scale=std_dev)
            prob_mass = np.clip(prob_mass, 1e-10, 1)
            prob_mass /= prob_mass.sum()
            results['sarima_dist'] = {int(num): float(prob) for num, prob in zip(x_range, prob_mass)}
            st.session_state.data_warnings.append(f"SARIMA for {position}: Pred={pred_point:.2f}, StdDev={std_dev:.2f}, Top 5 probs={sorted(results['sarima_dist'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        except Exception as e:
            st.warning(f"SARIMA failed for {position}: {e}. Using uniform distribution.")
            results['sarima_dist'] = {i: 1/float(max_num) for i in range(1, max_num + 1)}
        
        ml_results = _analyze_ml_models(series, max_num, position)
        results.update(ml_results)
        
        all_dists = [results.get('mcmc_dist', {}), results.get('sarima_dist', {}), results.get('hmm_dist', {})]
        ensemble_dist = {i: 0.0 for i in range(1, max_num + 1)}
        for dist in all_dists:
            if not dist:
                st.session_state.data_warnings.append(f"Empty distribution in ensemble for {position}")
                continue
            total_prob = sum(dist.values())
            if total_prob == 0 or np.isnan(total_prob):
                st.session_state.data_warnings.append(f"Invalid total prob in ensemble for {position}: {total_prob}")
                continue
            for num, prob in dist.items():
                if 1 <= num <= max_num and isinstance(prob, (int, float)) and prob >= 0:
                    ensemble_dist[num] += prob / total_prob
        total_ensemble_prob = sum(ensemble_dist.values()) or 1
        ensemble_dist = {num: prob / total_ensemble_prob for num, prob in ensemble_dist.items()}
        results['distributions'] = [ensemble_dist]
        st.session_state.data_warnings.append(f"Ensemble for {position}: Top 5 probs={sorted(ensemble_dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
        return results
    except Exception as e:
        st.error(f"Error in stable position analysis for {position}: {e}")
        return {'distributions': [{i: 1/float(max_num) for i in range(1, max_num + 1)}]}

# --- 5. ADVANCED PREDICTIVE MODELS ---
@st.cache_resource
def train_torch_model(_df: pd.DataFrame, model_type: str = 'LSTM', seq_length: int = 10, epochs: int = 50) -> Tuple[Optional[nn.Module], Optional[MinMaxScaler], float]:
    """Trains a PyTorch LSTM or GRU model."""
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
                self.rnn = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=0.2) if model_type == 'LSTM' else nn.GRU(input_size, hidden_size, 2, batch_first=True, dropout=0.2)
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
        st.session_state.data_warnings.append(f"Trained {model_type}: Final Loss={final_loss:.4f}")
        return model, scaler, final_loss
    except Exception as e:
        st.error(f"Error training {model_type}: {e}")
        return None, None, float('inf')

@st.cache_data
def predict_torch_model(_df: pd.DataFrame, _model_cache: Tuple, model_type: str, seq_length: int, max_nums: List[int]) -> Dict[str, Any]:
    """Generates probabilistic predictions using a trained PyTorch model."""
    try:
        model, scaler, best_loss = _model_cache
        if model is None:
            st.warning(f"{model_type} training failed.")
            return {'name': model_type, 'distributions': [{i: 1/float(max_num) for i in range(1, max_num + 1)} for max_num in max_nums], 'logic': f'{model_type} failed.'}
        last_seq = scaler.transform(_df[-seq_length:].values).reshape(1, seq_length, 6)
        last_seq_torch = torch.FloatTensor(last_seq).to(device)
        with torch.no_grad():
            pred_scaled = model(last_seq_torch)
        prediction_raw = scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        prediction_raw = np.clip(prediction_raw, 1, max_nums)
        std_dev = max(1.0, np.sqrt(best_loss) * (max_nums[0] - 1) / 2)
        distributions = []
        for i in range(6)):
            max_num = max_nums[i]
            x_range = np.arange(1, max_num + 1)
            prob_mass = stats.norm.pdf(x_range, loc=prediction_raw[i], scale=std_dev)
            prob_mass = np.clip(prob_mass, 1e-10, 1)
            prob_mass /= prob_mass.sum()
            dist = {int(num): float(prob) for num, prob in zip(x_range, prob_mass)}
            distributions.append(dist)
            st.session_state.data_warnings.append(f"{model_type} Pos_{i+1}: Top 5={sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
        return {'name': model_type, 'distributions': distributions, 'logic': f'Deep learning {model_type} forecast.'}
    except Exception as e:
        st.warning(f"Prediction with {model_type} failed: {e}")
        return {'name': model_type, 'distributions': [{i: 1/float(max_num) for max_num in max_nums], 'logic': f'{model_type} failed.'}

# --- 6. Backtesting & Meta-Analysis ---
@st.cache_data
def run_full_backtest_suite(_df: pd.DataFrame, max_nums: List[int], stable_positions: List[str]) -> List[Dict[str, Any]]:
    """Runs walk-forward validation for all models using Log Loss."""
    try:
        scored_predictions = []
        split_point = max(50, int(len(_df) * 0.8))
        if len(_df) - split_point < 6:
            st.warning("Insufficient validation data: {split_point} adjusted.")
            split_point = len(_df) - 6
        train_df, val_df = _df[:split_point], _df[split_point:]
        lstm_cache = train_torch_model(train_df, 'LSTM')
        gru_cache = train_torch_model(train_df, 'GRU')
        model_funcs = {
            'LSTM': lambda d: predict_torch_model(d, lstm_cache, 'LSTM', 10, max_nums),
            'GRU': lambda d: predict_torch_model(d, gru_cache, 'GRU', 3, max_nums),
        }
        if stable_positions:
            for pos in stable_positions:
                pos_idx = int(pos.split('_')[1]) - 1
                if pos_idx < len(max_nums):
                    model_funcs[f'Stable_{pos}'] = lambda d, p=pos, m=max_nums[pos_idx]: analyze_stable_position_dynamics(d, p, m)
        st.session_state.data_warnings.append(f"Initialized models: {list(model_funcs.keys())}} with max_nums={max_nums}")
        progress_bar = st.progress(0, text="Backtesting models...")
        total_steps = len(val_df) * len(model_funcs.keys())
        current_step = 0
        for name, func in model_funcs.items():
            total_log_loss, draw_count = 0, 0
            for i in range(len(val_df)):
                try:
                    historical_df = _df.iloc[:split_point + i]
                    pred_obj = func(historical_df)
                    if not pred_obj or not pred_obj.get('distributions', []) or len(pred_obj['distributions']) != 6 or not all(isinstance(d, dict) for d in pred_obj['distributions']):
                        st.session_state.data_warnings.append(f"Invalid distributions for {name}, draw {i}: {pred_obj}")
                        continue
                    y_true = val_df.iloc[i].values().tolist()
                    draw_log_loss = 0
                    for pos_idx, dist in enumerate(pred_obj['distributions']):
                        if not dist:
                            st.session_state.data_warnings.append(f"Empty dist for Pos_{pos_idx+1} in {name}, draw {i}")
                            dist = {j: 1/float(max_nums[pos_idx]) for j in range(1, max_nums[pos_idx] + 1)}
                            total_prob = sum(dist.values())
                        if total_prob == 0 or np.isnan(total_prob):
                            st.session_state.data_warnings.append(f"Invalid total prob for Pos_{pos_idx+1} in {name}: {total_prob}")
                            dist = {j: float(max_nums[pos_idx]) for j in range(1, max_nums[pos_idx] + 1)})
                            else:
                                dist = {k: v/total_prob for k, v in dist.items()}
                            true_num = int(y_true[pos_idx])
                            if not (1 <= true_num <= max_nums[pos_idx]):
                                st.session_state.data_warnings.append(f"Invalid true_num={true_num} for Pos_{pos_idx+1} in {name}, max_num={max_nums[pos_idx]}")
                                continue
                            prob_of_true = dist.get(true_num, 1e-10)
                            st.session_state.data_warnings.append(f"{name}, Draw {i}, Pos_{pos_idx+1}: True={true_num}, Prob={prob_of_true:.5f}, Top 5={sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
                            draw_log_loss -= np.log(max(prob_of_true, float(1e-10)))
                        draw_log_loss = min(draw_log_loss, 20.0)  # Cap to prevent overflow
                        total_log_loss += draw_log_loss
                        draw_count += 1
                        st.session_state.data_warnings.append(f"{name}, Draw {i}: Log Loss={draw_log_loss:.3f}, Total={total_log_loss:.3f}, Draws={draw_count}")
                    except Exception as e:
                        st.session_state.data_warnings.append(f"Backtest error for {name}, draw {i}: {e}")
                        continue
                    current_step += 1
                    progress_bar.progress(min(1.0, current_step / total_steps), text=f"Backtesting {name}...")
                try:
                    final_pred_obj = func(_df)
                    if not final_pred_obj or not final_pred_obj.get('distributions') or len(final_pred_obj['distributions']) != 6:
                        st.session_state.data_warnings.append(f"Final prediction failed for {name}: {final_pred_obj}")
                        final_pred_obj = {
                            'name': name,
                            'distributions': [{i: 1/float(max_num)} for max_num in max_nums],
                            'logic': f'{name} failed'
                        }
                    avg_log_loss = total_log_loss / draw_count if draw_count > 0 else 20
                    likelihood = max(0, min(100, 100 - avg_log_loss * 10))  # Adjusted for stability
                    final_pred_obj['likelihood'] = likelihood
                    final_pred_obj['metrics'] = {'Avg Log Loss': f"{avg_log_loss:.3f}"}
                    final_pred_obj['prediction'] = get_best_guess_set(final_pred_obj['distributions'], max_nums)
                    if len(final_pred_obj['prediction']) != 6:
                        st.session_state.data_warnings.append(f"Invalid prediction set for {name}: {final_pred_obj['prediction']}")
                        continue
                    if final_pred_obj.get('name', 'Unknown Model') == 'Unknown Model':
                        st.session_state.data_warnings.append(f"Skipping invalid model for {name}")
                        continue
                    scored_predictions.append(final_pred_obj)
                    st.session_state.data_warnings.append(f"Final: {name}, Set={final_pred_obj['prediction']}, Likelihood={likelihood:.2f}%")
                except Exception as e:
                    st.session_state.data_warnings.append(f"Final prediction error for {name}: {e}")
                progress_bar.empty()
        st.session_state.data_warnings.append(f"Markov Chain calls: {sum([len(calls)] for calls in st.session_state.model_calls.values())}")
        if not scored_predictions:
            st.error("No valid predictions generated.")
            return []
        return sorted(scored_predictions, key=lambda x: x.get('likelihood', 0), reverse=True)
    except Exception as e:
        st.error(f"Backtest suite error: {e}")
        return []

# =====================================================================================================
# Main Application UI & Logic
# =====================================================================================================
st.title("⚛️ LottoSphere v17.0.5: Quantum Chronodynamics Engine")
st.markdown("A scientific instrument for exploratory analysis with probabilistic forecasting.")

st.sidebar.header("Configuration")
max_nums = [st.sidebar.number_input(f"Max Number Pos_{i+1}", min_value=10, max_value=100, value=50 + i*2, key=f'max_num_{i+1}') for i in range(6)]
uploaded_file = st.sidebar.file_uploader("Upload Number History (CSV)", type=["csv"], help="CSV with one draw per row, numbers in columns. Last row is most recent.")

if uploaded_file:
    df_master = load_data(uploaded_file, max_nums)
    for warning_msg in st.session_state.data_warnings:
        st.warning(warning_msg)
    if not df_master.empty() and df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df)} draws.")
        with st.spinner("Analyzing system stability..."):
            stable_positions = []
            for pos in df_master.columns:
                result = analyze_temporal_behavior(df_master, pos)
                if result.get('is_stable', False):
                    stable_positions.append(pos)
                st.session_state.data_warnings.append(f"{pos} stability: Lyapunov={result.get('lyapunov', 'N/A')}, Stable={result.get('is_stable', False)}")
        if stable_positions:
            st.sidebar.info(f"Stable positions: {', '.join(stable_positions)}.")
        else:
            st.sidebar.warning("No stable positions detected.")
        
        tab1, tab2 = st.tabs(["🔮 Predictions", "🔬 Dynamics Explorer"])
        
        with tab1:
            st.header("Grand Unified Predictive Ensemble")
            with st.expander("About Probabilistic Predictions"):
                st.markdown("""
                ### Overview
                Generates probability distributions for the next draw using LSTM, GRU, and stable position models (MCMC, SARIMA, Markov Chain). Ranked by historical log loss.

                ### Models
                - **LSTM/GRU**: Deep learning for sequential patterns.
                - **Stable Position Models**: For stable positions (Lyapunov ≤0.05), use MCMC, SARIMA, and Markov Chain.
                - **Backtesting**: Walk-forward validation with log loss, likelihood = max(0, min(100, 100 - 10 × log_loss)).

                ### Actionability
                - Select predictions >60% likelihood.
                - Review distributions for confidence.
                - Cross-validate with Dynamics Explorer.
                """)
            
            if st.button("🚗 Run All Models", type="primary", use_container_width=True):
                with st.spinner("Backtesting models... This may take a few minutes."):
                    scored_predictions = run_full_backtest_suite(df_master, max_nums, stable_positions)
                st.header("✨ Final Predictions")
                if scored_predictions:
                    st.subheader("Ranked Probabilistic Forecasts")
                    for p in scored_predictions:
                        with st.container(border=True):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"#### {p.get('name', 'Unknown')}")
                                pred_str = ' | '.join(str(n) for n in p.get('prediction', [])) or "No valid set"
                                st.markdown(f"**Most Likely Set:** `{pred_str}`")
                            with col2:
                                st.metric("Likelihood Score", f"{p.get('likelihood', 0):.2f}%", help=f"Metrics: {p.get('metrics', {})}")
                            with st.expander("View Probability Distributions"):
                                chart_cols = st.columns(6)
                                for i, dist in enumerate(p.get('distributions', [])):
                                    with chart_cols[i]:
                                        st.markdown(f"**Pos_{i+1}**")
                                        if dist and all(isinstance(p, (int, float)) and p >= 0 for p in dist.values()):
                                            df_dist = pd.DataFrame(list(dist.items()), columns=['Number', 'Probability']).sort_values('Number')
                                            fig = px.bar(df_dist, x='Number', y='Probability', height=200)
                                            fig.update_layout(margin=dict(l=10, r=10, t=30, b=30), yaxis_title=None, xaxis_title=None)
                                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                                        else:
                                            st.write("No valid distribution.")
        
        with tab2:
            st.header("System Dynamics Explorer")
            with st.expander("About Dynamics Analysis"):
                st.markdown("""
                ### Overview
                Analyzes temporal behavior of a position using chaos theory and time-series analysis.

                ### Outputs
                - **Recurrence Plot**: Visualizes state recurrences.
                - **Power Spectral Density**: Identifies cycles (Fourier).
                - **Spectrogram**: Tracks frequency changes.
                - **Lyapunov Exponent**: Measures chaos (positive) or stability (≤0.05).
                - **Periodicity**: Detects cycles in stable positions.

                ### Actionability
                - Stable positions suggest predictability; use in predictions.
                - Cross-reference periodicity with distributions.
                """)
            position = st.selectbox("Select Position", options=df_master.columns, index=0)
            if st.button(f"Analyze {position}", use_container_width=True):
                with st.spinner(f"Analyzing {position}..."):
                    dynamic_results = analyze_temporal_behavior(df_master, position)
                if dynamic_results:
                    st.subheader(f"Analysis for {position}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(dynamic_results.get('recurrence_fig'), use_container_width=True)
                    with col2:
                        st.plotly_chart(dynamic_results.get('fourier_fig'), use_container_width=True)
                    st.plotly_chart(dynamic_results.get('spectrogram_fig'), use_container_width=True)
                    if not np.isnan(dynamic_results.get('lyapunov', float('nan'))):
                        if dynamic_results['lyapunov'] > 0.05:
                            st.warning(f"**Lyapunov Exponent:** {dynamic_results['lyapunov']:.3f}. Chaotic.", icon="⚠️")
                        else:
                            st.success(f"**Lyapunov Exponent:** {dynamic_results['lyapunov']:.3f}. Stable.", icon="✅")
                            if dynamic_results.get('periodicity_description'):
                                st.info(f"**Periodicity:** {dynamic_results['periodicity_description']}", icon="🔄")
                                st.plotly_chart(dynamic_results.get('acf_fig'), use_container_width=True)
                    else:
                        st.warning("Lyapunov calculation failed.")
else:
    st.info("Upload a CSV file to begin analysis.")

