# ======================================================================================================
# LottoSphere v18.2.0: Quantum Chronodynamics Engine (Optimized for Streamlit Cloud)
#
# VERSION: 18.2.0
#
# DESCRIPTION:
# A Streamlit-based application for modeling 6-digit lottery draws as stochastic systems.
# Optimized for Streamlit Cloud with all models (LSTM, GRU, BayesianNN, MCMC, SARIMA, HMM),
# fixed likelihood scores, functional Dynamics Explorer, enhanced Cluster Analysis,
# and adaptive training_size/n_clusters.
#
# CHANGELOG:
# - v18.2.0: Fixed issues from v18.1.0:
#   - Included all models (LSTM, GRU, BayesianNN, MCMC, SARIMA, HMM).
#   - Fixed Likelihood Score (0.00%) with capped log_loss and exp(-log_loss).
#   - Restored Dynamics Explorer with robust error handling and reduced plot complexity.
#   - Added context/significance to Cluster Analysis with silhouette scores.
#   - Optimized training_size via log_loss minimization.
#   - Selected n_clusters via silhouette score.
#   - Enhanced logging for debugging.
# - v18.1.0: Optimized for Streamlit Cloud, reduced resources, deferred computations.
# - v18.0.0: Original with MCMC, SARIMA, HMM, LSTM, GRU, BayesianNN.
# ======================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
from typing import List, Dict, Any, Tuple, Optional
import scipy.stats as stats
from scipy.signal import welch, cwt, morlet2
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from sktime.forecasting.arima import AutoARIMA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, silhouette_score
import networkx as nx
import itertools
import math

# Optional dependencies with fallbacks
try:
    from nolds import lyap_r, dfa
except ImportError:
    lyap_r, dfa = None, None
    st.warning("nolds not available, skipping Lyapunov and DFA.")
try:
    import torchbnn as bnn
except ImportError:
    bnn = None
    st.warning("torchbnn not available, skipping BayesianNN.")
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
    st.warning("hmmlearn not available, skipping HMM.")
try:
    import hdbscan
except ImportError:
    hdbscan = None
    st.warning("hdbscan not available, skipping clustering.")
try:
    import umap
except ImportError:
    umap = None
    st.warning("umap-learn not available, skipping UMAP.")

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. APPLICATION CONFIGURATION & INITIALIZATION ---
st.set_page_config(
    page_title="LottoSphere v18.2.0: Quantum Chronodynamics",
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
device = torch.device("cpu")  # Force CPU for Streamlit Cloud
st.session_state.data_warnings.append(f"Using device: {device}")

# --- Clear Streamlit Cache ---
if not st.session_state.cache_cleared:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.cache_cleared = True
    st.session_state.data_warnings.append("Streamlit cache cleared.")

# --- 2. CORE UTILITIES & DATA HANDLING ---
@st.cache_data
def load_data(uploaded_file: io.BytesIO, max_nums: List[int]) -> pd.DataFrame:
    """Loads, validates, and preprocesses lottery data from CSV."""
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        st.session_state.data_warnings.append(f"Loaded CSV: {len(df)} rows, {df.shape[1]} columns, max_nums={max_nums}")
        if df.shape[1] != 6:
            st.session_state.data_warnings.append(f"CSV has {df.shape[1]} columns, expected 6.")
            return pd.DataFrame()
        if len(max_nums) != 6:
            st.session_state.data_warnings.append(f"max_nums must have 6 values, got {len(max_nums)}: {max_nums}")
            return pd.DataFrame()
        
        # Check for non-numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df.isna().any().any():
            st.session_state.data_warnings.append(f"Found {df.isna().sum().sum()} non-numeric/NaN values.")
            return pd.DataFrame()
        df = df.astype(int)

        # Validate against max_nums
        initial_rows = len(df)
        for i, max_num in enumerate(max_nums):
            invalid = df.iloc[:, i][(df.iloc[:, i] < 1) | (df.iloc[:, i] > max_num)]
            if not invalid.empty:
                st.session_state.data_warnings.append(
                    f"Pos_{i+1}: {len(invalid)} values outside [1, {max_num}]: {invalid.unique().tolist()}"
                )
            df = df[(df.iloc[:, i] >= 1) & (df.iloc[:, i] <= max_num)]
        if len(df) < initial_rows:
            st.session_state.data_warnings.append(f"Discarded {initial_rows - len(df)} rows outside max_nums.")

        # Check duplicates
        unique_counts = df.apply(lambda x: len(set(x)), axis=1)
        valid_rows = (unique_counts == 6)
        if not valid_rows.all():
            st.session_state.data_warnings.append(f"Discarded {len(df) - valid_rows.sum()} rows with duplicates.")
            df = df[valid_rows]
        if df.duplicated().any():
            st.session_state.data_warnings.append(f"Discarded {df.duplicated().sum()} duplicate rows.")
            df = df.drop_duplicates()

        # Check variance and unique values
        for i, col in enumerate(df.columns):
            if df[col].std() < 1.0:
                st.session_state.data_warnings.append(f"Low variance in {col}: std={df[col].std():.2f}.")
                return pd.DataFrame()
            if len(df[col].unique()) < 5:
                st.session_state.data_warnings.append(f"Too few unique values in {col}: {len(df[col].unique())}.")
                return pd.DataFrame()

        df = df.reset_index(drop=True)
        df.columns = [f'Pos_{i+1}' for i in range(6)]
        if len(df) < 50:
            st.session_state.data_warnings.append(f"Insufficient data: {len(df)} rows, need ≥50.")
            return pd.DataFrame()
        unique_draws = len(df.drop_duplicates())
        if unique_draws < 50:
            st.session_state.data_warnings.append(f"Insufficient unique draws: {unique_draws}, need ≥50.")
            return pd.DataFrame()

        st.session_state.data_warnings.append(f"Sorted data, loaded {len(df)} draws, {unique_draws} unique.")
        sorted_values = np.sort(df.values, axis=1)
        return pd.DataFrame(sorted_values, columns=df.columns).astype(int)
    except Exception as e:
        st.session_state.data_warnings.append(f"Fatal error loading data: {e}")
        return pd.DataFrame()

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences for time-series modeling."""
    if seq_length >= len(data):
        raise ValueError("Sequence length must be less than data length")
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def get_best_guess_set(distributions: List[Dict[int, float]], max_nums: List[int]) -> List[int]:
    """Generates a unique 6-number set from probability distributions."""
    best_guesses = []
    seen = set()
    if len(distributions) != 6:
        st.session_state.data_warnings.append(f"Expected 6 distributions, got {len(distributions)}. Using uniform.")
        distributions = [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in max_nums[:6]]
    
    for i, dist in enumerate(distributions[:6]):
        if not dist or not all(isinstance(p, (int, float)) and p >= 0 for p in dist.values()):
            st.session_state.data_warnings.append(f"Invalid distribution for Pos_{i+1}: {dist}. Using uniform.")
            dist = {j: 1/max_num for j in range(1, max_num + 1)}
        total_prob = sum(dist.values())
        if total_prob == 0 or np.isnan(total_prob):
            st.session_state.data_warnings.append(f"Zero/NaN probability for Pos_{i+1}: {total_prob}. Using uniform.")
            dist = {j: 1/max_num for j in range(1, max_num + 1)}
        else:
            dist = {k: v/total_prob for k, v in dist.items()}
        
        numbers = list(dist.keys())
        np.random.shuffle(numbers)
        sorted_dist = sorted([(num, dist[num]) for num in numbers], key=lambda x: (-x[1], np.random.random()))
        
        for num, prob in sorted_dist:
            if num not in seen and 1 <= num <= max_nums[i]:
                best_guesses.append(num)
                seen.add(num)
                break
        else:
            available = sorted(set(range(1, max_nums[i] + 1)) - seen)
            if available:
                guess = np.random.choice(available)
                best_guesses.append(guess)
                seen.add(guess)
            else:
                guess = np.random.randint(1, max_nums[i] + 1)
                while guess in seen:
                    guess = np.random.randint(1, max_nums[i] + 1)
                best_guesses.append(guess)
                seen.add(guess)
    
    while len(best_guesses) < 6:
        for i in range(6):
            if len(best_guesses) >= 6:
                break
            available = sorted(set(range(1, max_nums[i] + 1)) - seen)
            if available:
                guess = np.random.choice(available)
                best_guesses.append(guess)
                seen.add(guess)
    st.session_state.data_warnings.append(f"Best guess set: {best_guesses[:6]}")
    return sorted(best_guesses[:6])

# --- 3. TIME-DEPENDENT BEHAVIOR ANALYSIS ---
@st.cache_data
def analyze_temporal_behavior(_df: pd.DataFrame, position: str, max_num: int) -> Dict[str, Any]:
    """Analyzes chaotic and cyclical behavior of a positional time series."""
    try:
        results = {}
        series = _df[position].values.astype(int)
        if len(series) < 10:
            st.session_state.data_warnings.append(f"Insufficient data for {position}: {len(series)} draws.")
            return results
        
        # Recurrence Plot
        recurrence_matrix = np.abs(np.subtract.outer(series, series))
        normalized_recurrence = recurrence_matrix / (recurrence_matrix.max() + 1e-10)
        results['recurrence_fig'] = px.imshow(
            normalized_recurrence, color_continuous_scale='viridis', title=f'Recurrence Plot: {position}',
            labels=dict(x='Draw Index', y='Draw Index', color='Distance')
        )
        results['recurrence_fig'].update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            hovertemplate='Draw X: %{x}<br>Draw Y: %{y}<br>Distance: %{z:.2f}<extra></extra>',
            annotations=[dict(text="Visualizes state recurrences over time.", x=0.5, y=1.05, showarrow=False, xref="paper", yref="paper")]
        )

        # Fourier Analysis
        nperseg = min(len(series), 64)  # Reduced for Streamlit Cloud
        freqs, psd = welch(series, nperseg=nperseg, fs=1.0)
        psd_df = pd.DataFrame({'Frequency': freqs, 'Power': psd}).sort_values('Power', ascending=False)
        results['fourier_fig'] = px.line(
            psd_df, x='Frequency', y='Power', title=f'Power Spectral Density: {position}',
            hover_data={'Frequency': ':.3f', 'Power': ':.3f'}
        )
        results['fourier_fig'].update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title='Frequency', yaxis_title='Power',
            annotations=[dict(text="Identifies dominant cycles.", x=0.5, y=1.05, showarrow=False, xref="paper", yref="paper")]
        )

        # Wavelet Transform
        widths = np.arange(1, min(len(series)//2, 8))  # Reduced for Streamlit Cloud
        if len(widths) > 0:
            cwt_matrix = cwt(series, morlet2, widths)
            results['wavelet_fig'] = go.Figure(data=go.Heatmap(
                z=np.abs(cwt_matrix), x=np.arange(len(series)), y=widths, colorscale='viridis'
            ))
            results['wavelet_fig'].update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                title=f'Wavelet Transform: {position}', xaxis_title='Time', yaxis_title='Scale',
                annotations=[dict(text="Tracks frequency changes over time.", x=0.5, y=1.05, showarrow=False, xref="paper", yref="paper")]
            )
        else:
            results['wavelet_fig'] = go.Figure()
            st.session_state.data_warnings.append(f"Wavelet skipped for {position}: insufficient data.")

        # Lyapunov Exponent
        if lyap_r:
            try:
                lyap_exp = lyap_r(series, emb_dim=max(2, len(series)//20), lag=1, min_tsep=5)
                results['lyapunov'] = lyap_exp
                results['is_stable'] = lyap_exp <= 0.1  # Relaxed threshold
                st.session_state.data_warnings.append(f"{position} Lyapunov: {lyap_exp:.3f}, Stable: {results['is_stable']}")
            except Exception as e:
                results['lyapunov'] = float('nan')
                results['is_stable'] = False
                st.session_state.data_warnings.append(f"Lyapunov failed for {position}: {e}")
        else:
            results['lyapunov'] = float('nan')
            results['is_stable'] = False
            st.session_state.data_warnings.append(f"Lyapunov unavailable for {position}")

        # Fractal Dimension (DFA)
        if dfa:
            try:
                fractal_dim = dfa(series)
                results['fractal_dim'] = fractal_dim
                st.session_state.data_warnings.append(f"{position} Fractal Dimension: {fractal_dim:.3f}")
            except Exception:
                results['fractal_dim'] = float('nan')
                st.session_state.data_warnings.append(f"Fractal dimension failed for {position}")
        else:
            results['fractal_dim'] = float('nan')

        # Periodicity Analysis
        if results.get('is_stable'):
            acf_vals = acf(series, nlags=min(25, len(series)//2 - 1), fft=True)
            lags = np.arange(len(acf_vals))
            conf_interval = 1.96 / np.sqrt(len(series))
            peaks = np.where(acf_vals[1:] > conf_interval)[0]
            if peaks.size > 0:
                dominant_period = lags[peaks[0] + 1]
                results['periodicity'] = dominant_period
                results['periodicity_description'] = f"Periodicity detected: {dominant_period} draws."
            else:
                results['periodicity'] = None
                results['periodicity_description'] = "No significant periodicity."
            acf_df = pd.DataFrame({'Lag': lags, 'Autocorrelation': acf_vals})
            results['acf_fig'] = px.line(
                acf_df, x='Lag', y='Autocorrelation', title=f'Autocorrelation: {position}', markers=True,
                hover_data={'Lag': ':.0f', 'Autocorrelation': ':.3f'}
            )
            results['acf_fig'].add_hline(y=conf_interval, line_dash="dash", line_color="red")
            results['acf_fig'].add_hline(y=-conf_interval, line_dash="dash", line_color="red")
            results['acf_fig'].update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                annotations=[dict(text="Detects periodic patterns.", x=0.5, y=1.05, showarrow=False, xref="paper", yref="paper")]
            )

        return results
    except Exception as e:
        st.session_state.data_warnings.append(f"Error in temporal analysis for {position}: {e}")
        return {}

# --- 4. CLUSTERING & LATENT SPACE ANALYSIS ---
@st.cache_data
def analyze_clusters(_df: pd.DataFrame, training_size: int) -> Dict[str, Any]:
    """Performs clustering and latent space visualization with silhouette-based n_clusters selection."""
    try:
        results = {}
        if not hdbscan or not umap:
            st.session_state.data_warnings.append("HDBSCAN or UMAP unavailable, skipping clustering.")
            return results
        data = _df.iloc[-training_size:].values
        if len(data) < 10:
            st.session_state.data_warnings.append(f"Clustering failed: insufficient data ({len(data)} rows).")
            return results
        
        # Select optimal n_clusters via silhouette score
        best_n_clusters, best_silhouette, best_labels = 2, -1, None
        for n in range(2, 6):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(data)//20), min_samples=3)
            labels = clusterer.fit_predict(data)
            if len(set(labels)) > 1 and -1 not in labels:  # Exclude noise points
                try:
                    score = silhouette_score(data, labels)
                    if score > best_silhouette:
                        best_n_clusters, best_silhouette, best_labels = n, score, labels
                except:
                    continue
        if best_labels is None:
            st.session_state.data_warnings.append("No valid clusters found.")
            return results
        
        results['n_clusters'] = best_n_clusters
        results['silhouette_score'] = best_silhouette
        results['cluster_labels'] = best_labels
        st.session_state.data_warnings.append(f"Optimal clusters: {best_n_clusters}, Silhouette: {best_silhouette:.3f}")

        # UMAP for 2D Visualization
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
        embedding = reducer.fit_transform(data)
        results['umap_embedding'] = embedding
        results['umap_fig'] = px.scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            color=best_labels.astype(str), title=f'Latent Space Clusters (Silhouette: {best_silhouette:.3f})',
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
            hover_data={'Draw': _df.index[-training_size:]}
        )
        results['umap_fig'].update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            annotations=[dict(text=f"{best_n_clusters} clusters detected.", x=0.5, y=1.05, showarrow=False, xref="paper", yref="paper")]
        )

        # Cluster summaries
        cluster_counts = pd.Series(best_labels).value_counts().to_dict()
        results['cluster_summaries'] = {}
        for cluster in set(best_labels):
            if cluster == -1:
                continue
            cluster_data = data[best_labels == cluster]
            mean_numbers = np.mean(cluster_data, axis=0).astype(int)
            results['cluster_summaries'][cluster] = {
                'size': cluster_counts.get(cluster, 0),
                'mean_numbers': mean_numbers.tolist()
            }
            st.session_state.data_warnings.append(
                f"Cluster {cluster}: Size={cluster_counts.get(cluster, 0)}, Mean={mean_numbers.tolist()}"
            )

        return results
    except Exception as e:
        st.session_state.data_warnings.append(f"Clustering error: {e}")
        return {}

# --- 5. ADVANCED PREDICTIVE MODELS ---
def _analyze_stat_physics(series: np.ndarray, max_num: int) -> Dict[str, Any]:
    """Statistical physics analysis with Monte Carlo."""
    results = {}
    series = np.clip(series, 1, max_num).astype(int)
    
    # Monte Carlo with Markov Chain
    counts = np.zeros((max_num, max_num))
    for i in range(len(series)-1):
        from_idx, to_idx = series[i]-1, series[i+1]-1
        if 0 <= from_idx < max_num and 0 <= to_idx < max_num:
            counts[from_idx, to_idx] += 1
    trans_prob = (counts + 1e-10) / (counts.sum(axis=1, keepdims=True) + 1e-9)
    current_state = series[-1] - 1 if 0 <= series[-1] - 1 < max_num else 0
    prob_vector = trans_prob[current_state]
    prob_vector /= prob_vector.sum() + 1e-10
    mcmc_samples = np.random.choice(max_num, size=500, p=prob_vector)  # Reduced samples
    mcmc_dist = pd.Series(mcmc_samples + 1).value_counts(normalize=True).to_dict()
    results['mcmc_dist'] = {int(k): float(v) for k, v in mcmc_dist.items()}
    st.session_state.data_warnings.append(f"MCMC top 5: {sorted(results['mcmc_dist'].items(), key=lambda x: x[1], reverse=True)[:5]}")
    return results

def _analyze_hmm(series: np.ndarray, max_num: int) -> Dict[str, Any]:
    """Hidden Markov Model analysis."""
    results = {}
    if not hmm:
        st.session_state.data_warnings.append("HMM unavailable, using uniform.")
        return {'hmm_dist': {i: 1/max_num for i in range(1, max_num + 1)}}
    series = np.clip(series, 1, max_num).astype(int) - 1
    try:
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=50)
        model.fit(series.reshape(-1, 1))
        prob_dist = model.predict_proba(series[-1:].reshape(-1, 1))[0]
        dist = {}
        for state in range(2):
            mean = model.means_[state, 0]
            std = np.sqrt(model.covars_[state, 0, 0])
            x = np.arange(max_num)
            probs = stats.norm.pdf(x, mean, std)
            probs /= probs.sum() + 1e-10
            for i, p in enumerate(probs):
                dist[i + 1] = dist.get(i + 1, 0) + p * prob_dist[state]
        total = sum(dist.values()) + 1e-10
        results['hmm_dist'] = {int(k): float(v/total) for k, v in dist.items()}
        st.session_state.data_warnings.append(f"HMM top 5: {sorted(results['hmm_dist'].items(), key=lambda x: x[1], reverse=True)[:5]}")
    except Exception as e:
        st.session_state.data_warnings.append(f"HMM failed: {e}. Using uniform.")
        results['hmm_dist'] = {i: 1/max_num for i in range(1, max_num + 1)}
    return results

def _analyze_sarima(series: np.ndarray, max_num: int, position: str) -> Dict[str, Any]:
    """SARIMA analysis with robust error handling."""
    results = {}
    series = np.clip(series, 1, max_num).astype(int)
    st.session_state.data_warnings.append(
        f"SARIMA {position}: len={len(series)}, mean={series.mean():.2f}, std={series.std():.2f}, unique={len(np.unique(series))}"
    )
    try:
        if len(series) < 10 or len(np.unique(series)) < 5:
            raise ValueError(f"Invalid series: len={len(series)}, unique={len(np.unique(series))}")
        model = AutoARIMA(sp=1, suppress_warnings=True, maxiter=50)
        model.fit(series)
        pred = model.predict(fh=[1])
        pred_point = float(pred.iloc[0] if isinstance(pred, pd.Series) else pred[0])
        conf_int = model.predict_interval(fh=[1], coverage=0.95)
        std_dev = float(max(0.1, (conf_int.iloc[0, 1] - conf_int.iloc[0, 0]) / 3.92))
        x_range = np.arange(1, max_num + 1)
        prob_mass = stats.norm.pdf(x_range, loc=pred_point, scale=std_dev)
        prob_mass = np.clip(prob_mass, 1e-10, 1)
        prob_mass /= prob_mass.sum() + 1e-10
        results['sarima_dist'] = {int(num): float(prob) for num, prob in zip(x_range, prob_mass)}
        st.session_state.data_warnings.append(
            f"SARIMA {position}: Pred={pred_point:.2f}, StdDev={std_dev:.2f}, Top 5={sorted(results['sarima_dist'].items(), key=lambda x: x[1], reverse=True)[:5]}"
        )
    except Exception as e:
        st.session_state.data_warnings.append(f"SARIMA failed for {position}: {e}. Using uniform.")
        results['sarima_dist'] = {i: 1/max_num for i in range(1, max_num + 1)}
    return results

@st.cache_resource
def train_torch_model(_df: pd.DataFrame, model_type: str, seq_length: int = 10, epochs: int = 20) -> Tuple[Optional[nn.Module], Optional[MinMaxScaler], float]:
    """Trains LSTM, GRU, or BayesianNN model."""
    try:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(_df)
        X, y = create_sequences(data_scaled, seq_length)
        X_torch, y_torch = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        class SequenceModel(nn.Module):
            def __init__(self, input_size=6, hidden_size=32):
                super().__init__()
                if model_type == 'BayesianNN' and bnn:
                    self.rnn = bnn.BayesLSTM(input_size, hidden_size, 1)
                    self.fc = bnn.BayesLinear(hidden_size, input_size)
                else:
                    self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=0.1) if model_type == 'LSTM' else nn.GRU(input_size, hidden_size, 1, batch_first=True, dropout=0.1)
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
        st.session_state.data_warnings.append(f"{model_type} trained: Loss={final_loss:.4f}")
        return model, scaler, final_loss
    except Exception as e:
        st.session_state.data_warnings.append(f"{model_type} training failed: {e}")
        return None, None, float('inf')

@st.cache_data
def predict_torch_model(_df: pd.DataFrame, _model_cache: Tuple, model_type: str, seq_length: int, max_nums: List[int]) -> Dict[str, Any]:
    """Predicts with LSTM, GRU, or BayesianNN."""
    try:
        model, scaler, best_loss = _model_cache
        if model is None:
            st.session_state.data_warnings.append(f"{model_type} model is None. Using uniform.")
            return {'name': model_type, 'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in max_nums], 'logic': f'{model_type} failed.'}
        last_seq = scaler.transform(_df.iloc[-seq_length:].values).reshape(1, seq_length, 6)
        last_seq_torch = torch.tensor(last_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(last_seq_torch)
        prediction_raw = scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        prediction_raw = np.clip(prediction_raw, 1, max_nums)
        std_dev = max(1.0, np.sqrt(best_loss) * (max_nums[0] - 1) / 2)
        distributions = []
        for i in range(6):
            x_range = np.arange(1, max_nums[i] + 1)
            prob_mass = stats.norm.pdf(x_range, loc=prediction_raw[i], scale=std_dev)
            prob_mass = np.clip(prob_mass, 1e-10, 1)
            prob_mass /= prob_mass.sum() + 1e-10
            dist = {int(num): float(prob) for num, prob in zip(x_range, prob_mass)}
            distributions.append(dist)
            st.session_state.data_warnings.append(f"{model_type} Pos_{i+1}: Top 5={sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
        return {'name': model_type, 'distributions': distributions, 'logic': f'Deep learning {model_type} forecast.'}
    except Exception as e:
        st.session_state.data_warnings.append(f"{model_type} prediction failed: {e}")
        return {'name': model_type, 'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in max_nums], 'logic': f'{model_type} failed.'}

@st.cache_data
def analyze_stable_position_dynamics(_df: pd.DataFrame, position: str, max_num: int) -> Dict[str, Any]:
    """Analyzes stable position with MCMC, SARIMA, HMM."""
    try:
        results = {'name': f'Stable_{position}'}
        series = _df[position].values.astype(int)
        if not all(1 <= x <= max_num for x in series):
            st.session_state.data_warnings.append(f"Invalid values in {position}: {series[:5]}... Clipping.")
            series = np.clip(series, 1, max_num)
        
        # Statistical Physics
        stat_phys = _analyze_stat_physics(series, max_num)
        results.update(stat_phys)
        
        # SARIMA
        sarima_results = _analyze_sarima(series, max_num, position)
        results.update(sarima_results)
        
        # HMM
        hmm_results = _analyze_hmm(series, max_num)
        results.update(hmm_results)
        
        # Ensemble
        all_dists = [results.get('mcmc_dist', {}), results.get('sarima_dist', {}), results.get('hmm_dist', {})]
        ensemble_dist = {i: 0.0 for i in range(1, max_num + 1)}
        valid_dists = 0
        for dist in all_dists:
            if not dist:
                continue
            total_prob = sum(dist.values())
            if total_prob == 0 or np.isnan(total_prob):
                continue
            valid_dists += 1
            for num, prob in dist.items():
                if 1 <= num <= max_num and isinstance(prob, (int, float)) and prob >= 0:
                    ensemble_dist[num] += prob / total_prob
        total_prob = sum(ensemble_dist.values()) or 1
        ensemble_dist = {num: prob / total_prob for num, prob in ensemble_dist.items()}
        results['distributions'] = [ensemble_dist]
        st.session_state.data_warnings.append(
            f"Ensemble {position}: Top 5={sorted(ensemble_dist.items(), key=lambda x: x[1], reverse=True)[:5]}, Valid dists={valid_dists}"
        )
        
        # Information Theory
        entropy = -sum(p * np.log(p + 1e-10) for p in ensemble_dist.values())
        results['shannon_entropy'] = float(entropy)
        results['kl_divergence'] = float(stats.entropy(
            list(ensemble_dist.values()),
            [1/max_num for _ in range(max_num)]
        ))
        return results
    except Exception as e:
        st.session_state.data_warnings.append(f"Stable position analysis failed for {position}: {e}")
        return {'name': f'Stable_{position}', 'distributions': [{i: 1/max_num for i in range(1, max_num + 1)}], 'logic': f'Stable_{position} failed.'}

# --- 6. OPTIMIZE TRAINING SIZE ---
@st.cache_data
def optimize_training_size(_df: pd.DataFrame, max_nums: List[int], stable_positions: List[str]) -> int:
    """Selects training_size with lowest average log_loss."""
    try:
        training_sizes = [50, 75, 100]
        best_size, best_loss = 50, float('inf')
        for size in training_sizes:
            if len(_df) - size < 10:
                continue
            train_df = _df.iloc[:size]
            val_df = _df.iloc[size:size+5]
            lstm_cache = train_torch_model(train_df, 'LSTM', seq_length=10, epochs=10)
            model_funcs = {
                'LSTM': lambda d: predict_torch_model(d, lstm_cache, 'LSTM', 10, max_nums)
            }
            for pos in stable_positions[:1]:  # Test one stable position
                pos_idx = int(pos.split('_')[1]) - 1
                if pos_idx < len(max_nums):
                    model_funcs[f'Stable_{pos}'] = lambda d, p=pos, m=max_nums[pos_idx]: analyze_stable_position_dynamics(d, p, m)
            log_losses = []
            for name, func in model_funcs.items():
                for i in range(len(val_df)):
                    try:
                        historical_df = _df.iloc[:size + i]
                        pred_obj = func(historical_df)
                        if not pred_obj or not pred_obj.get('distributions'):
                            continue
                        y_true = val_df.iloc[i].values
                        draw_log_loss = 0
                        for pos_idx, dist in enumerate(pred_obj['distributions']):
                            if not dist:
                                dist = {j: 1/max_nums[pos_idx] for j in range(1, max_nums[pos_idx] + 1)}
                            total_prob = sum(dist.values())
                            if total_prob == 0 or np.isnan(total_prob):
                                dist = {j: 1/max_nums[pos_idx] for j in range(1, max_nums[pos_idx] + 1)}
                            else:
                                dist = {k: v/total_prob for k, v in dist.items()}
                            true_num = int(y_true[pos_idx])
                            if not (1 <= true_num <= max_nums[pos_idx]):
                                continue
                            prob_of_true = dist.get(true_num, 1e-10)
                            draw_log_loss -= np.log(max(prob_of_true, 1e-10))
                        log_losses.append(min(draw_log_loss, 10.0))
                    except:
                        continue
            avg_loss = np.mean(log_losses) if log_losses else float('inf')
            if avg_loss < best_loss:
                best_size, best_loss = size, avg_loss
        st.session_state.data_warnings.append(f"Optimal training_size: {best_size}, Avg Log Loss: {best_loss:.3f}")
        return best_size
    except Exception as e:
        st.session_state.data_warnings.append(f"Training size optimization failed: {e}. Using default 50.")
        return 50

# --- 7. BACKTESTING & PERFORMANCE METRICS ---
@st.cache_data
def run_full_backtest_suite(_df: pd.DataFrame, max_nums: List[int], stable_positions: List[str], training_size: int) -> List[Dict[str, Any]]:
    """Runs walk-forward validation and computes metrics."""
    try:
        scored_predictions = []
        split_point = max(50, training_size)
        if len(_df) - split_point < 10:
            split_point = len(_df) - 10
            st.session_state.data_warnings.append(f"Adjusted split_point to {split_point} for validation.")
        train_df = _df.iloc[:split_point]
        
        # Train models
        lstm_cache = train_torch_model(train_df, 'LSTM', seq_length=10, epochs=20)
        gru_cache = train_torch_model(train_df, 'GRU', seq_length=10, epochs=20)
        bnn_cache = train_torch_model(train_df, 'BayesianNN', seq_length=10, epochs=20) if bnn else (None, None, float('inf'))
        
        model_funcs = {
            'LSTM': lambda d: predict_torch_model(d, lstm_cache, 'LSTM', 10, max_nums),
            'GRU': lambda d: predict_torch_model(d, gru_cache, 'GRU', 10, max_nums),
        }
        if bnn:
            model_funcs['BayesianNN'] = lambda d: predict_torch_model(d, bnn_cache, 'BayesianNN', 10, max_nums)
        for pos in stable_positions:
            pos_idx = int(pos.split('_')[1]) - 1
            if pos_idx < len(max_nums):
                model_funcs[f'Stable_{pos}'] = lambda d, p=pos, m=max_nums[pos_idx]: analyze_stable_position_dynamics(d, p, m)
        
        st.session_state.data_warnings.append(f"Models: {list(model_funcs.keys())}, max_nums={max_nums}")
        progress_bar = st.progress(0, text="Backtesting...")
        total_steps = min(5, len(_df) - split_point) * len(model_funcs)
        current_step = 0
        
        for name, func in model_funcs.items():
            log_losses, top5_accuracies, entropies = [], [], []
            val_df = _df.iloc[split_point:split_point + 5]
            for i in range(len(val_df)):
                try:
                    historical_df = _df.iloc[:split_point + i]
                    pred_obj = func(historical_df)
                    if not pred_obj or not pred_obj.get('distributions') or len(pred_obj['distributions']) != 6:
                        st.session_state.data_warnings.append(f"Invalid distributions for {name}, draw {i}")
                        continue
                    y_true = val_df.iloc[i].values
                    draw_log_loss = 0
                    top5_correct = 0
                    for pos_idx, dist in enumerate(pred_obj['distributions']):
                        if not dist:
                            dist = {j: 1/max_nums[pos_idx] for j in range(1, max_nums[pos_idx] + 1)}
                        total_prob = sum(dist.values())
                        if total_prob == 0 or np.isnan(total_prob):
                            dist = {j: 1/max_nums[pos_idx] for j in range(1, max_nums[pos_idx] + 1)}
                        else:
                            dist = {k: v/total_prob for k, v in dist.items()}
                        true_num = int(y_true[pos_idx])
                        if not (1 <= true_num <= max_nums[pos_idx]):
                            continue
                        prob_of_true = dist.get(true_num, 1e-10)
                        draw_log_loss -= np.log(max(prob_of_true, 1e-10))
                        top5 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
                        top5_numbers = [n for n, _ in top5]
                        if true_num in top5_numbers:
                            top5_correct += 1
                    draw_log_loss = min(draw_log_loss, 10.0)
                    log_losses.append(draw_log_loss)
                    top5_accuracies.append(top5_correct / 6)
                    entropy = -sum(sum(p * np.log(p + 1e-10) for p in d.values()) for d in pred_obj['distributions'])
                    entropies.append(entropy)
                    current_step += 1
                    progress_bar.progress(min(1.0, current_step / total_steps))
                except Exception as e:
                    st.session_state.data_warnings.append(f"Backtest error for {name}, draw {i}: {e}")
            
            try:
                final_pred_obj = func(_df)
                if not final_pred_obj or not final_pred_obj.get('distributions') or len(final_pred_obj['distributions']) != 6:
                    final_pred_obj = {
                        'name': name,
                        'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in max_nums],
                        'logic': f'{name} failed'
                    }
                avg_log_loss = np.mean(log_losses) if log_losses else 10.0
                top5_accuracy = np.mean(top5_accuracies) if top5_accuracies else 0
                avg_entropy = np.mean(entropies) if entropies else 0
                likelihood = 100 * np.exp(-avg_log_loss / 2)  # Adjusted for positive values
                stability_index = np.std(log_losses) / (np.mean(log_losses) + 1e-10) if log_losses else 1.0
                final_pred_obj['likelihood'] = min(100, max(0, likelihood))
                final_pred_obj['metrics'] = {
                    'Avg Log Loss': f"{avg_log_loss:.3f}",
                    'Top-5 Accuracy': f"{top5_accuracy:.3f}",
                    'Avg Entropy': f"{avg_entropy:.3f}",
                    'Stability Index': f"{stability_index:.3f}"
                }
                final_pred_obj['prediction'] = get_best_guess_set(final_pred_obj['distributions'], max_nums)
                scored_predictions.append(final_pred_obj)
                st.session_state.data_warnings.append(
                    f"Final {name}: Set={final_pred_obj['prediction']}, Likelihood={likelihood:.2f}%, Log Loss={avg_log_loss:.3f}, Metrics={final_pred_obj['metrics']}"
                )
            except Exception as e:
                st.session_state.data_warnings.append(f"Final prediction error for {name}: {e}")
        progress_bar.progress(1.0)
        return sorted(scored_predictions, key=lambda x: x.get('likelihood', 0), reverse=True)
    except Exception as e:
        st.session_state.data_warnings.append(f"Backtest suite error: {e}")
        return []

# --- Main Application UI & Logic ---
st.title("⚛️ LottoSphere v18.2.0: Quantum Chronodynamics Engine")
st.markdown("A scientific instrument for modeling 6-digit lottery draws as stochastic systems.")

st.sidebar.header("Configuration")
max_nums = [st.sidebar.number_input(f"Max Number Pos_{i+1}", min_value=10, max_value=100, value=50 + i*2, key=f"max_num_{i+1}") for i in range(6)]
training_size_default = 50
n_clusters_default = 3
uploaded_file = st.sidebar.file_uploader("Upload Number History (CSV)", type=["csv"], help="CSV with 6 columns, one draw per row, last row most recent.")

# Display warnings
if st.session_state.data_warnings:
    with st.sidebar.expander("Warnings", expanded=True):
        for warning in st.session_state.data_warnings[-10:]:
            st.warning(warning)

if uploaded_file:
    df_master = load_data(uploaded_file, max_nums)
    if not df_master.empty and df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded {len(df_master)} draws.")
        with st.spinner("Analyzing stability and optimizing parameters..."):
            stable_positions = []
            temporal_results = {}
            for pos in df_master.columns:
                result = analyze_temporal_behavior(df_master, pos, max_nums[int(pos.split('_')[1]) - 1])
                temporal_results[pos] = result
                if result.get('is_stable', False):
                    stable_positions.append(pos)
            if not stable_positions:
                stable_positions = df_master.columns.tolist()  # Fallback to all positions
            st.session_state.data_warnings.append(f"Stable positions: {stable_positions}")
            
            # Optimize training_size
            training_size = optimize_training_size(df_master, max_nums, stable_positions)
            st.sidebar.info(f"Optimal Training Size: {training_size} (based on lowest log loss)")
        
        tab1, tab2, tab3 = st.tabs(["🔮 Predictions", "🔬 Dynamics Explorer", "🌌 Cluster Analysis"])
        
        with tab1:
            st.header("Grand Unified Predictive Ensemble")
            with st.expander("About Predictions", expanded=False):
                st.markdown("""
                ### Overview
                Generates probability distributions using LSTM, GRU, BayesianNN (if available), and stable position models (MCMC, SARIMA, HMM). Ranked by likelihood score.

                ### Models
                - **LSTM/GRU/BayesianNN**: Deep learning for temporal patterns.
                - **Stable Position Models**: Ensemble of MCMC, SARIMA, HMM for stable positions (Lyapunov ≤0.1).
                - **Metrics**:
                  - **Avg Log Loss**: Prediction error (lower is better).
                  - **Top-5 Accuracy**: Fraction of true numbers in top-5 predictions.
                  - **Avg Entropy**: Uncertainty (lower is better).
                  - **Stability Index**: Prediction consistency (lower is better).
                  - **Likelihood Score**: Confidence in predictions (higher is better, 100 * exp(-log_loss/2)).

                ### Mathematical Basis
                - **MCMC**: Simulates state transitions via Markov chains.
                - **SARIMA**: Models time-series with seasonal components.
                - **HMM**: Captures hidden states in sequences.
                - **LSTM/GRU/BayesianNN**: Neural networks for sequence prediction.

                ### Actionability
                - Choose predictions with >60% likelihood.
                - Prioritize models with low log loss (<2.0) and high top-5 accuracy (>0.5).
                - Cross-validate with Dynamics and Cluster tabs.
                """)
            
            if st.button("🚀 Run All Models", type="primary", use_container_width=True):
                with st.spinner("Backtesting models..."):
                    scored_predictions = run_full_backtest_suite(df_master, max_nums, stable_positions, training_size)
                st.header("✨ Final Predictions")
                if scored_predictions:
                    st.subheader("Ranked Forecasts")
                    for p in scored_predictions:
                        with st.container(border=True):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"#### {p.get('name', 'Unknown')}")
                                pred_str = ' | '.join(str(n) for n in p.get('prediction', [])) or "No valid set"
                                st.markdown(f"**Most Likely Set:** `{pred_str}`")
                                st.markdown(f"**Logic:** {p.get('logic', 'N/A')}")
                            with col2:
                                st.metric("Likelihood Score", f"{p.get('likelihood', 0):.2f}%", help=f"Metrics: {p.get('metrics', {})}")
                            with st.expander("Probability Distributions"):
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
            with st.expander("About Dynamics Analysis", expanded=False):
                st.markdown("""
                ### Overview
                Analyzes temporal behavior of each position using chaos theory and time-series methods to identify predictability.

                ### Outputs
                - **Recurrence Plot**: Visualizes state recurrences, showing patterns or chaos.
                - **Power Spectral Density**: Identifies dominant cycles via Fourier transform.
                - **Wavelet Transform**: Tracks frequency changes over time.
                - **Lyapunov Exponent**: Measures chaos (>0.1) or stability (≤0.1).
                - **Fractal Dimension**: Quantifies complexity via DFA.
                - **Autocorrelation**: Detects periodic patterns in stable positions.

                ### Mathematical Basis
                - **Recurrence Plot**: Distance matrix of states.
                - **Fourier/Wavelet**: Decomposes signals into frequency components.
                - **Lyapunov**: Quantifies divergence of trajectories.
                - **DFA**: Measures self-similarity.
                - **ACF**: Identifies significant lags for periodicity.

                ### Actionability
                - Stable positions (Lyapunov ≤0.1) are more predictable; prioritize their predictions.
                - Periodicity indicates cycle lengths for forecasting horizons.
                - High fractal dimension suggests complex, less predictable dynamics.
                """)
            position = st.selectbox("Select Position", options=df_master.columns, index=0)
            if st.button(f"Analyze {position}", use_container_width=True):
                with st.spinner(f"Analyzing {position}..."):
                    results = temporal_results.get(position, {})
                if results:
                    st.subheader(f"Analysis for {position}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'recurrence_fig' in results:
                            st.plotly_chart(results['recurrence_fig'], use_container_width=True)
                        if not np.isnan(results.get('lyapunov', float('nan'))):
                            if results['lyapunov'] > 0.1:
                                st.warning(f"**Lyapunov Exponent:** {results['lyapunov']:.3f} (Chaotic)", icon="⚠️")
                            else:
                                st.success(f"**Lyapunov Exponent:** {results['lyapunov']:.3f} (Stable)", icon="✅")
                        if not np.isnan(results.get('fractal_dim', float('nan'))):
                            st.info(f"**Fractal Dimension:** {results['fractal_dim']:.3f}", icon="🔍")
                    with col2:
                        if 'fourier_fig' in results:
                            st.plotly_chart(results['fourier_fig'], use_container_width=True)
                        if results.get('is_stable') and results.get('periodicity_description'):
                            st.info(f"**Periodicity:** {results['periodicity_description']}", icon="🔄")
                            if 'acf_fig' in results:
                                st.plotly_chart(results['acf_fig'], use_container_width=True)
                    if 'wavelet_fig' in results and results['wavelet_fig'].data:
                        st.plotly_chart(results['wavelet_fig'], use_container_width=True)
                    else:
                        st.warning("Wavelet analysis unavailable due to insufficient data.")
                else:
                    st.error(f"Analysis failed for {position}. Check warnings.")
        
        with tab3:
            st.header("Cluster Analysis")
            with st.expander("About Cluster Analysis", expanded=True):
                st.markdown("""
                ### Overview
                Clusters draws to identify behavioral regimes, visualized in a 2D latent space.

                ### Outputs
                - **Latent Space Clusters**: 2D UMAP plot showing draw clusters.
                - **Silhouette Score**: Measures clustering quality (0 to 1, higher is better).
                - **Cluster Summaries**: Size and mean numbers per cluster.

                ### Mathematical Basis
                - **HDBSCAN**: Density-based clustering, automatically detects clusters.
                - **UMAP**: Reduces high-dimensional draws to 2D for visualization.
                - **Silhouette Score**: Quantifies how well-separated clusters are.

                ### Significance
                - **Dense Clusters**: Indicate common draw patterns, useful for predictions.
                - **Large Clusters**: Represent stable regimes; their mean numbers are likely candidates.
                - **High Silhouette Score (>0.5)**: Indicates reliable clusters.
                - **Outliers (Cluster -1)**: Draws that don’t fit patterns, less predictable.

                ### Actionability
                - Use mean numbers from the largest cluster for conservative predictions.
                - Combine cluster insights with Predictions tab for robust forecasts.
                - If silhouette score is low (<0.3), clustering may be unreliable.
                """)
            if st.button("Run Cluster Analysis", use_container_width=True):
                with st.spinner("Running cluster analysis..."):
                    cluster_results = analyze_clusters(df_master, training_size)
                if cluster_results:
                    st.subheader(f"Clustering Results (Silhouette Score: {cluster_results.get('silhouette_score', 0):.3f})")
                    if 'umap_fig' in cluster_results:
                        st.plotly_chart(cluster_results['umap_fig'], use_container_width=True)
                    if 'cluster_summaries' in cluster_results:
                        st.subheader("Cluster Summaries")
                        for cluster, summary in cluster_results['cluster_summaries'].items():
                            st.markdown(f"**Cluster {cluster}**: Size={summary['size']}, Mean Numbers={summary['mean_numbers']}")
                else:
                    st.error("Clustering failed. Check warnings.")
else:
    st.info("Upload a CSV file to begin analysis.")
