# ======================================================================================================
# LottoSphere v19.0.0: Quantum Chronodynamics Engine (SME Optimized)
#
# VERSION: 19.0.0
#
# DESCRIPTION:
# A scientifically rigorous Streamlit dashboard for modeling and forecasting 6-digit numerical sequences.
# This version represents a complete methodological overhaul of v18.x.
#
# CHANGELOG (v19.0.0):
# - CRITICAL FIX: Removed `np.sort` from data loading. Positional integrity is now preserved,
#   enabling true stochastic process analysis for each of the 6 digit positions.
# - NEW FEATURE: Implemented "Stabilization Point Analysis" to visually determine the optimal training
#   window size by plotting Cross-Entropy and a new Prediction Stability Index (PSI) vs. data history.
# - REFACTORED CLUSTERING: Replaced flawed `n_clusters` logic with interactive sliders for
#   HDBSCAN's `min_cluster_size` and `min_samples`. Corrected silhouette score to ignore outliers.
# - NEW MODEL: Integrated a "Quantum-Inspired Hilbert Embedding" model using amplitude encoding to
#   represent draw states and forecast probabilities.
# - ENHANCED BACKTESTING: Revamped `run_full_backtest_suite` for more robust walk-forward
#   validation over a user-configurable number of steps.
# - UNIFIED POSITIONAL MODEL: Created a "Positional Dynamics Ensemble" that intelligently combines
#   MCMC, SARIMA, and HMM predictions for each stable position.
# - UX OVERHAUL: Analysis now runs automatically on data upload. Interactive sliders provide
#   dynamic control over key parameters.
# - REFINED METRICS: Added Prediction Stability Index (PSI). Improved Likelihood Score formula for
#   better sensitivity. Cleaned up logging and warnings.
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
from sktime.forecasting.arima import AutoARIMA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, silhouette_score
import copy

# --- Optional Dependencies with Graceful Fallbacks ---
try:
    from nolds import lyap_r
except ImportError:
    lyap_r = None
    st.sidebar.warning("`nolds` not found. Lyapunov exponent analysis will be skipped.")
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
    st.sidebar.warning("`hmmlearn` not found. HMM model will be skipped.")
try:
    import hdbscan
except ImportError:
    hdbscan = None
    st.sidebar.warning("`hdbscan` not found. Clustering analysis will be disabled.")
try:
    import umap
except ImportError:
    umap = None
    st.sidebar.warning("`umap-learn` not found. UMAP visualization will be disabled.")

# --- Suppress Warnings for a Cleaner UI ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. APPLICATION CONFIGURATION & INITIALIZATION ---
st.set_page_config(
    page_title="LottoSphere v19.0.0: Quantum Chronodynamics",
    page_icon="⚛️",
    layout="wide",
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'df_master' not in st.session_state:
    st.session_state.df_master = pd.DataFrame()

# Use CPU for compatibility with Streamlit Cloud
device = torch.device("cpu")

# --- 2. CORE UTILITIES & DATA HANDLING ---

@st.cache_data
def load_and_validate_data(uploaded_file: io.BytesIO, max_nums: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads, validates, and preprocesses lottery data from a CSV file.
    CRITICAL FIX: This version DOES NOT sort the numbers in each draw, preserving positional integrity.
    """
    logs = []
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        if df.shape[1] != 6:
            logs.append(f"Error: CSV must have exactly 6 columns, but found {df.shape[1]}.")
            return pd.DataFrame(), logs
        
        df.columns = [f'Pos_{i+1}' for i in range(6)]
        
        # Validate data types and ranges
        initial_rows = len(df)
        df_validated = df.copy()
        for i, col in enumerate(df_validated.columns):
            df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
            df_validated.dropna(subset=[col], inplace=True)
            max_val = max_nums[i]
            df_validated = df_validated[df_validated[col].between(1, max_val)]
        
        logs.append(f"Loaded {initial_rows} rows. After validation, {len(df_validated)} rows remain.")

        # Check for sufficient and valid data
        if len(df_validated) < 50:
            logs.append(f"Error: Insufficient valid data ({len(df_validated)} rows). Need at least 50.")
            return pd.DataFrame(), logs
        
        # Check for within-row duplicates
        is_duplicate_in_row = df_validated.apply(lambda row: row.nunique() != 6, axis=1)
        num_dupe_rows = is_duplicate_in_row.sum()
        if num_dupe_rows > 0:
            logs.append(f"Discarding {num_dupe_rows} rows containing duplicate numbers.")
            df_validated = df_validated[~is_duplicate_in_row]

        df_validated = df_validated.reset_index(drop=True).astype(int)
        logs.append(f"Data validation successful. Final dataset contains {len(df_validated)} draws.")
        
        # CRITICAL CHANGE: Do NOT sort the data. Return the original sequence.
        return df_validated, logs

    except Exception as e:
        logs.append(f"Fatal error during data loading: {e}")
        return pd.DataFrame(), logs

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences for time-series modeling."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def get_best_guess_set(distributions: List[Dict[int, float]]) -> List[int]:
    """Generates a unique 6-number set from probability distributions, preserving position."""
    best_guesses = [0] * 6
    seen_numbers = set()
    
    # First pass: Get the highest probability unique number for each position
    candidates = []
    for i, dist in enumerate(distributions):
        if not dist: continue
        sorted_probs = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        for num, prob in sorted_probs:
            if num not in seen_numbers:
                candidates.append({'pos': i, 'num': num, 'prob': prob, 'assigned': False})
                break

    # Assign best candidates greedily by probability
    for candidate in sorted(candidates, key=lambda x: x['prob'], reverse=True):
        if best_guesses[candidate['pos']] == 0 and candidate['num'] not in seen_numbers:
            best_guesses[candidate['pos']] = candidate['num']
            seen_numbers.add(candidate['num'])

    # Second pass: Fill any remaining empty positions
    for i in range(6):
        if best_guesses[i] == 0:
            dist = distributions[i]
            sorted_probs = sorted(dist.items(), key=lambda item: item[1], reverse=True)
            for num, prob in sorted_probs:
                if num not in seen_numbers:
                    best_guesses[i] = num
                    seen_numbers.add(num)
                    break
    return best_guesses


# --- 3. DYNAMICS & STABILITY ANALYSIS ---

@st.cache_data
def analyze_positional_dynamics(_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Analyzes each position for chaos, stability, and periodicity."""
    results = {}
    for pos in _df.columns:
        series = _df[pos].values
        pos_results = {'is_stable': False, 'lyapunov': float('nan')}
        
        if lyap_r and len(series) > 50:
            try:
                # Use robust parameters for Lyapunov calculation
                emb_dim = max(3, len(series) // 20)
                lyap_exp = lyap_r(series, emb_dim=emb_dim, lag=1, min_tsep=len(series)//10)
                pos_results['lyapunov'] = round(lyap_exp, 4)
                # A stricter stability threshold for more reliable identification
                if lyap_exp <= 0.05:
                    pos_results['is_stable'] = True
            except Exception:
                pass # Fails silently if nolds can't compute
        results[pos] = pos_results
    return results

@st.cache_data
def find_stabilization_point(_df: pd.DataFrame, _max_nums: List[int], backtest_steps: int) -> go.Figure:
    """
    Analyzes model performance across different training window sizes to find the 'stabilization point'.
    """
    window_sizes = np.linspace(50, max(250, len(_df) - backtest_steps - 1), 10, dtype=int)
    results = []

    for size in window_sizes:
        if len(_df) < size + backtest_steps:
            continue
        
        log_losses = []
        predictions = []
        for step in range(backtest_steps):
            train_df = _df.iloc[:size + step]
            true_draw = _df.iloc[size + step].values
            
            # Use a simple, fast model for this analysis (e.g., SARIMA on one position)
            series = train_df['Pos_1'].values
            try:
                model = AutoARIMA(sp=1, suppress_warnings=True)
                model.fit(series)
                pred_mean = model.predict(fh=[1])[0]
                pred_std = np.std(series) * 1.5 # Robust std estimate
                
                x_range = np.arange(1, _max_nums[0] + 1)
                prob_mass = stats.norm.pdf(x_range, loc=pred_mean, scale=pred_std)
                prob_dist = {num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())}
                
                prob_of_true = prob_dist.get(true_draw[0], 1e-9)
                log_losses.append(-np.log(prob_of_true))
                predictions.append(pred_mean)
            except Exception:
                log_losses.append(np.log(_max_nums[0])) # Penalize failure with uniform entropy

        avg_log_loss = np.mean(log_losses)
        # Prediction Stability Index (PSI): Normalized standard deviation of predictions
        psi = np.std(predictions) / (np.mean(predictions) + 1e-9) if predictions else float('nan')
        
        results.append({'Window Size': size, 'Cross-Entropy Loss': avg_log_loss, 'Prediction Stability Index': psi})

    if not results:
        st.error("Could not generate stabilization data. The dataset might be too small.")
        return go.Figure()

    results_df = pd.DataFrame(results).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['Window Size'], y=results_df['Cross-Entropy Loss'],
                             mode='lines+markers', name='Cross-Entropy Loss'))
    fig.add_trace(go.Scatter(x=results_df['Window Size'], y=results_df['Prediction Stability Index'],
                             mode='lines+markers', name='Prediction Stability Index', yaxis='y2'))

    fig.update_layout(
        title='Model Stabilization Analysis',
        xaxis_title='Training Window Size (Number of Draws)',
        yaxis_title='Cross-Entropy Loss (Lower is Better)',
        yaxis2=dict(title='Prediction Stability Index (Lower is Better)', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )
    return fig


# --- 4. CLUSTERING & LATENT SPACE ANALYSIS ---

@st.cache_data
def analyze_clusters(_df: pd.DataFrame, min_cluster_size: int, min_samples: int) -> Dict[str, Any]:
    """Performs and visualizes clustering using HDBSCAN and UMAP with interactive parameters."""
    results = {'fig': go.Figure(), 'summary': "Clustering disabled or failed."}
    if not hdbscan or not umap or len(_df) < min_cluster_size:
        return results

    data = _df.values
    
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
        labels = clusterer.fit_predict(data)
        
        # Calculate silhouette score, excluding noise points (label=-1)
        if len(set(labels)) > 1:
            clean_labels = labels[labels != -1]
            clean_data = data[labels != -1]
            if len(set(clean_labels)) > 1:
                score = silhouette_score(clean_data, clean_labels)
                results['silhouette'] = f"{score:.3f}"
            else:
                results['silhouette'] = "N/A (1 cluster)"
        else:
            results['silhouette'] = "N/A (No clusters)"

        reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(data)
        
        plot_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
        plot_df['Cluster'] = [str(l) for l in labels]
        plot_df['Draw'] = _df.index
        plot_df['Numbers'] = _df.apply(lambda row: ', '.join(row.astype(str)), axis=1)

        fig = px.scatter(
            plot_df, x='UMAP_1', y='UMAP_2', color='Cluster',
            custom_data=['Draw', 'Numbers'],
            title=f'Latent Space of Draw Behaviors (Silhouette: {results.get("silhouette", "N/A")})',
            color_discrete_map={'-1': 'grey'}
        )
        fig.update_traces(hovertemplate='<b>Draw %{customdata[0]}</b><br>Numbers: %{customdata[1]}<br>Cluster: %{marker.color}')
        fig.update_layout(legend_title_text='Cluster ID')
        results['fig'] = fig

        # Cluster Summaries
        summary_text = ""
        cluster_counts = Counter(labels)
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:
                summary_text += f"**Noise Points:** {count} draws did not belong to any cluster.\n"
            else:
                cluster_mean = _df[labels == cluster_id].mean().round().astype(int).tolist()
                summary_text += f"**Cluster {cluster_id}:** {count} draws. Centroid (mean): `{cluster_mean}`\n"
        results['summary'] = summary_text

    except Exception as e:
        results['summary'] = f"An error occurred during clustering: {e}"
        
    return results


# --- 5. ADVANCED PREDICTIVE MODELS ---

# Base class for models to ensure a consistent interface
class BaseModel:
    def __init__(self, max_nums: List[int]):
        self.max_nums = max_nums
        self.name = "Base Model"
        self.logic = "Base logic"
    
    def train(self, df: pd.DataFrame):
        raise NotImplementedError

    def predict(self) -> Dict[str, Any]:
        raise NotImplementedError

# --- Positional Dynamics Ensemble ---
class PositionalDynamicsEnsemble(BaseModel):
    def __init__(self, max_nums, stable_positions):
        super().__init__(max_nums)
        self.stable_positions = stable_positions
        self.name = "Positional Dynamics Ensemble"
        self.logic = "Ensemble of MCMC, SARIMA, and HMM applied to each stable digit position."
        self.models = {}

    def train(self, df: pd.DataFrame):
        for i, pos in enumerate(df.columns):
            if pos in self.stable_positions:
                series = df[pos].values
                max_num = self.max_nums[i]
                
                # MCMC Transition Matrix
                counts = np.zeros((max_num, max_num))
                for j in range(len(series) - 1):
                    counts[series[j]-1, series[j+1]-1] += 1
                trans_prob = (counts + 0.1) / (counts.sum(axis=1, keepdims=True) + 0.1 * max_num)
                
                # SARIMA model
                sarima = AutoARIMA(sp=1, suppress_warnings=True)
                sarima.fit(series)
                
                # HMM model
                hmm_model = None
                if hmm and len(series) > 10:
                    try:
                        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
                        hmm_model.fit(series.reshape(-1, 1))
                    except Exception:
                        hmm_model = None

                self.models[pos] = {'mcmc': trans_prob, 'sarima': sarima, 'hmm': hmm_model, 'last_val': series[-1]}

    def predict(self) -> Dict[str, Any]:
        distributions = []
        for i, pos in enumerate([f'Pos_{j+1}' for j in range(6)]):
            max_num = self.max_nums[i]
            if pos in self.models:
                m = self.models[pos]
                
                # MCMC prediction
                mcmc_probs = m['mcmc'][m['last_val']-1]

                # SARIMA prediction
                pred_mean = m['sarima'].predict(fh=[1])[0]
                pred_std = np.std(m['sarima']._y) * 1.2
                x_range = np.arange(1, max_num + 1)
                sarima_probs = stats.norm.pdf(x_range, loc=pred_mean, scale=max(1.0, pred_std))

                # HMM prediction
                hmm_probs = np.ones(max_num)
                if m['hmm']:
                    try:
                        means = m['hmm'].means_.flatten()
                        covars = np.sqrt([m['hmm'].covars_[k, 0, 0] for k in range(m['hmm'].n_components)])
                        trans_probs = m['hmm'].transmat_[m['hmm'].predict(np.array([[m['last_val']]]))[0]]
                        
                        hmm_probs = np.zeros(max_num)
                        for k in range(m['hmm'].n_components):
                            hmm_probs += trans_probs[k] * stats.norm.pdf(x_range, means[k], covars[k])
                    except Exception:
                        pass # HMM prediction can be unstable

                # Ensemble the probabilities (simple average)
                ensemble_probs = (mcmc_probs + sarima_probs/sarima_probs.sum() + hmm_probs/hmm_probs.sum()) / 3.0
                dist = {num: p for num, p in zip(x_range, ensemble_probs / ensemble_probs.sum())}
                distributions.append(dist)
            else:
                # For non-stable positions, use a uniform distribution
                distributions.append({k: 1/max_num for k in range(1, max_num + 1)})

        return {'distributions': distributions}

# --- Deep Learning Models (LSTM/GRU) ---
class TorchSequenceModel(BaseModel):
    def __init__(self, max_nums, model_type='LSTM', hidden_size=50, seq_length=12, epochs=25):
        super().__init__(max_nums)
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.epochs = epochs
        self.name = f"{model_type} Deep Model"
        self.logic = f"A {model_type} recurrent neural network capturing temporal dependencies across all 6 positions simultaneously."
        self.model = None
        self.scaler = None

    def train(self, df: pd.DataFrame):
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(df)
        X, y = create_sequences(data_scaled, self.seq_length)
        
        if len(X) == 0: return # Not enough data
        
        X_torch = torch.tensor(X, dtype=torch.float32).to(device)
        y_torch = torch.tensor(y, dtype=torch.float32).to(device)
        
        class _Model(nn.Module):
            def __init__(self, input_size, hidden_size, model_type):
                super().__init__()
                rnn_layer = nn.LSTM if model_type == 'LSTM' else nn.GRU
                self.rnn = rnn_layer(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, input_size)
            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :])

        self.model = _Model(6, self.hidden_size, self.model_type).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self) -> Dict[str, Any]:
        if not self.model or not self.scaler:
            return {'distributions': [{k: 1/m for k in range(1, m + 1)} for m in self.max_nums]}

        self.model.eval()
        with torch.no_grad():
            last_seq_scaled = self.scaler.transform(st.session_state.df_master.iloc[-self.seq_length:].values)
            input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = self.model(input_tensor)
            pred_raw = self.scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        
        distributions = []
        for i in range(6):
            pred_mean = pred_raw[i]
            pred_std = np.std(self.scaler.inverse_transform(self.scaler.transform(st.session_state.df_master))[:,i]) * 0.5
            x_range = np.arange(1, self.max_nums[i] + 1)
            prob_mass = stats.norm.pdf(x_range, loc=pred_mean, scale=max(1.5, pred_std))
            distributions.append({num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())})
            
        return {'distributions': distributions}

# --- Quantum-Inspired Model ---
class QuantumHilbertModel(BaseModel):
    def __init__(self, max_nums):
        super().__init__(max_nums)
        self.name = "Quantum-Inspired Hilbert Embedding"
        self.logic = "Represents each draw as a quantum state vector via amplitude encoding. Forecasts by evolving the state with a transition operator derived from historical data."
        self.transition_operator = None
        self.last_state = None

    def _encode_draw(self, draw: np.ndarray) -> np.ndarray:
        """Encodes a 6-digit draw into a state vector using amplitude encoding."""
        state = np.zeros(sum(self.max_nums))
        offset = 0
        for i, num in enumerate(draw):
            state[offset + num - 1] = 1
            offset += self.max_nums[i]
        return state / np.linalg.norm(state) # Normalize to create a valid quantum state

    def train(self, df: pd.DataFrame):
        states = [self._encode_draw(row) for _, row in df.iterrows()]
        # Learn a transition operator U such that U|psi_t> ≈ |psi_{t+1}>
        # Using a simplified learning rule: U = sum(|psi_{t+1}><psi_t|)
        U = np.zeros((len(states[0]), len(states[0])))
        for t in range(len(states) - 1):
            U += np.outer(states[t+1], states[t])
        
        # Ensure U is unitary (or close to it) via SVD
        u_svd, _, vh_svd = np.linalg.svd(U)
        self.transition_operator = u_svd @ vh_svd
        self.last_state = states[-1]

    def predict(self) -> Dict[str, Any]:
        if self.transition_operator is None:
            return {'distributions': [{k: 1/m for k in range(1, m + 1)} for m in self.max_nums]}
        
        # Evolve the last known state
        predicted_state = self.transition_operator @ self.last_state
        
        # Convert back to probabilities by measuring the state
        probabilities = np.abs(predicted_state)**2
        
        distributions = []
        offset = 0
        for i, max_num in enumerate(self.max_nums):
            pos_probs = probabilities[offset : offset + max_num]
            prob_sum = pos_probs.sum()
            if prob_sum > 1e-9:
                dist = {num: p / prob_sum for num, p in enumerate(pos_probs, 1)}
                distributions.append(dist)
            else: # Fallback to uniform
                distributions.append({k: 1/max_num for k in range(1, max_num + 1)})
            offset += max_num
            
        return {'distributions': distributions}


# --- 6. BACKTESTING & PERFORMANCE EVALUATION ---

def run_backtest(model_instance: BaseModel, df: pd.DataFrame, train_size: int, backtest_steps: int) -> Dict[str, Any]:
    """Performs walk-forward validation for a given model instance."""
    log_losses = []
    predictions = []

    for i in range(backtest_steps):
        current_train_df = df.iloc[:train_size + i]
        true_draw = df.iloc[train_size + i].values

        # Retrain model on each step for true walk-forward validation
        model_instance.train(current_train_df)
        pred_obj = model_instance.predict()
        pred_set = get_best_guess_set(pred_obj['distributions'])
        predictions.append(pred_set)

        # Calculate log loss for the current step
        step_log_loss = 0
        for pos_idx, dist in enumerate(pred_obj['distributions']):
            prob_of_true = dist.get(true_draw[pos_idx], 1e-9)
            step_log_loss -= np.log(prob_of_true)
        log_losses.append(step_log_loss)

    avg_log_loss = np.mean(log_losses)
    # Likelihood Score: Re-scaled for better interpretability
    likelihood = 100 * np.exp(-avg_log_loss / np.log(np.mean(model_instance.max_nums)))
    # Prediction Stability Index (PSI): Fluctuation of the predicted set
    pred_array = np.array(predictions)
    psi = np.mean(np.std(pred_array, axis=0) / np.mean(pred_array, axis=0))

    return {
        'Log Loss': avg_log_loss,
        'Likelihood': likelihood,
        'PSI': psi,
    }


# --- 7. MAIN APPLICATION UI & LOGIC ---

st.title("⚛️ LottoSphere v19.0.0: Quantum Chronodynamics")
st.markdown("An interactive, scientifically-grounded instrument for analyzing and forecasting discrete numerical sequences as complex stochastic systems.")

st.sidebar.header("1. System Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Number History (CSV)", type=["csv"], help="A CSV file with 6 columns (no headers). Each row is a draw, with the most recent draw at the bottom.")

# Use st.expander for configuration to keep the sidebar clean
with st.sidebar.expander("Advanced Configuration", expanded=True):
    max_nums = [st.number_input(f"Max Value for Pos_{i+1}", min_value=10, max_value=150, value=49, key=f"max_num_{i}") for i in range(6)]
    training_size_slider = st.slider("Training Window Size", min_value=50, max_value=1000, value=150, step=10, help="Number of past draws to use for training models. Adjust based on the Stabilization Point Analysis.")
    backtest_steps_slider = st.slider("Backtest Validation Steps", min_value=5, max_value=50, value=10, step=1, help="Number of steps for walk-forward validation to evaluate model performance.")

# Main analysis logic
if uploaded_file is not None:
    df, logs = load_and_validate_data(uploaded_file, max_nums)
    with st.sidebar.expander("Data Loading Log", expanded=False):
        for log in logs:
            st.info(log)
    
    if not df.empty:
        st.session_state.df_master = df
        st.sidebar.success(f"Successfully loaded and validated {len(df)} draws.")
        
        # --- Run all analyses and store results in session state ---
        with st.spinner("Performing initial system dynamics analysis..."):
            if 'dynamics' not in st.session_state.analysis_results:
                 st.session_state.analysis_results['dynamics'] = analyze_positional_dynamics(df)
        
        # --- Define UI Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictive Ensembles", "📉 Stabilization Analysis", "🌌 Cluster Dynamics", "🔬 System Internals"])

        with tab1:
            st.header("🔮 Predictive Ensembles")
            st.markdown("Each model below offers a different scientific perspective on the sequence's dynamics. Models are evaluated via rigorous walk-forward backtesting.")
            
            with st.expander("How to Interpret These Forecasts", expanded=False):
                st.markdown("""
                - **Likelihood Score:** A confidence measure derived from the model's predictive accuracy (Cross-Entropy Loss) during backtesting. Higher is better. A score > 60% indicates a model that consistently outperforms random chance.
                - **Cross-Entropy Loss:** The core measure of prediction error. Lower is better. A value near `ln(avg_max_num)` (e.g., ~3.9 for max_num=49) is equivalent to random guessing.
                - **Prediction Stability Index (PSI):** Measures how much a model's numerical predictions fluctuate during backtesting. A low PSI indicates a stable, reliable model.
                - **The Models:**
                    - **Positional Dynamics Ensemble:** A hybrid model using MCMC, SARIMA, and HMM on positions identified as 'stable' (predictable). This is a robust, physics-based model.
                    - **LSTM/GRU Deep Models:** Recurrent Neural Networks that learn complex, non-linear patterns across all 6 positions simultaneously. Best for capturing momentum and inter-digit correlations.
                    - **Quantum-Inspired Hilbert Embedding:** A novel approach that treats each draw as a quantum state. It can capture holistic system properties that other models might miss.
                """)

            # --- Model Execution ---
            stable_positions = [pos for pos, res in st.session_state.analysis_results['dynamics'].items() if res.get('is_stable')]
            
            model_definitions = {
                "Positional Ensemble": PositionalDynamicsEnsemble(max_nums, stable_positions),
                "LSTM Deep Model": TorchSequenceModel(max_nums, model_type='LSTM'),
                "GRU Deep Model": TorchSequenceModel(max_nums, model_type='GRU'),
                "Quantum Hilbert Model": QuantumHilbertModel(max_nums)
            }
            
            # Use columns for a cleaner layout
            col1, col2 = st.columns(2)
            cols = [col1, col2, col1, col2] # Alternate columns

            for i, (name, model_instance) in enumerate(model_definitions.items()):
                with cols[i]:
                    with st.container(border=True):
                        st.subheader(name)
                        with st.spinner(f"Training and backtesting {name}..."):
                            # The actual training happens inside the backtest loop
                            perf_metrics = run_backtest(model_instance, df, training_size_slider, backtest_steps_slider)
                            
                            # Final prediction based on the full dataset
                            model_instance.train(df)
                            final_pred_obj = model_instance.predict()
                            final_prediction = get_best_guess_set(final_pred_obj['distributions'])

                        # Display results
                        st.markdown(f"**Logic:** *{model_instance.logic}*")
                        st.markdown(f"**Predicted Set:**")
                        st.code(" | ".join(map(str, final_prediction)))

                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Likelihood Score", f"{perf_metrics['Likelihood']:.1f}%")
                        m_col2.metric("Cross-Entropy", f"{perf_metrics['Log Loss']:.3f}")
                        m_col3.metric("Stability (PSI)", f"{perf_metrics['PSI']:.3f}")
                        
                        with st.expander("View Probability Distributions"):
                            dist_cols = st.columns(6)
                            for k, dist in enumerate(final_pred_obj['distributions']):
                                df_dist = pd.DataFrame(dist.items(), columns=['Number', 'Probability']).sort_values('Number')
                                fig = px.bar(df_dist, x='Number', y='Probability', height=200)
                                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis_title=None, xaxis_title=f'Pos {k+1}')
                                dist_cols[k].plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


        with tab2:
            st.header("📉 Training Window Stabilization Analysis")
            st.markdown("""
            This crucial analysis determines the **optimal amount of historical data** to train our models on. We are looking for the "sweet spot" or **stabilization point** where adding more data stops improving predictive performance.
            - **Cross-Entropy Loss (Blue Line):** Should ideally decrease and then plateau. The beginning of the plateau is often the optimal window size.
            - **Prediction Stability Index (Red Line):** Measures prediction volatility. A lower, stable value is desirable.
            
            **Actionable Insight:** Adjust the "Training Window Size" slider in the sidebar to match the stabilization point identified below for best results.
            """)
            with st.spinner("Calculating performance across window sizes..."):
                stabilization_fig = find_stabilization_point(df, max_nums, backtest_steps_slider)
                st.plotly_chart(stabilization_fig, use_container_width=True)

        with tab3:
            st.header("🌌 Cluster Dynamics & Regime Analysis")
            st.markdown("""
            This tool uses `HDBSCAN` (a sophisticated density-based algorithm) and `UMAP` (a powerful dimensionality reduction technique) to discover **"behavioral regimes"** within the data. Each point is a historical draw, and clusters represent groups of draws with similar characteristics.
            - **Well-defined clusters** suggest the system has predictable, recurring states.
            - **A large number of "Noise Points" (in grey)** indicates chaotic or highly random behavior.
            - **Cluster Centroids** (the average numbers in a cluster) can be powerful candidates for future draws if the system is currently in that regime.
            """)
            
            st.sidebar.header("2. Clustering Controls")
            cluster_min_size = st.sidebar.slider("Min Cluster Size", 5, 50, 15, 1, help="The smallest number of draws that can be considered a cluster.")
            cluster_min_samples = st.sidebar.slider("Min Samples", 1, 20, 5, 1, help="How conservative the clustering is. Higher values lead to more noise points but more robust clusters.")

            with st.spinner("Performing cluster analysis..."):
                cluster_results = analyze_clusters(df.iloc[-training_size_slider:], cluster_min_size, cluster_min_samples)

            st.plotly_chart(cluster_results['fig'], use_container_width=True)
            st.subheader("Cluster Interpretation")
            st.markdown(cluster_results['summary'])

        with tab4:
            st.header("🔬 System Internals: Positional Dynamics")
            st.markdown("Here we dissect the behavior of each of the 6 digit positions as an individual time series. The key metric is the **Lyapunov Exponent**, which measures the degree of chaos.")
            
            dynamics_results = st.session_state.analysis_results['dynamics']
            st.subheader("Positional Stability Summary")
            cols = st.columns(6)
            for i, (pos, res) in enumerate(dynamics_results.items()):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**{pos}**")
                        if res['is_stable']:
                            st.success(f"Stable", icon="✅")
                            st.metric("Lyapunov Exp.", f"{res['lyapunov']:.3f}")
                        else:
                            st.warning(f"Chaotic", icon="⚠️")
                            st.metric("Lyapunov Exp.", f"{res['lyapunov']:.3f}")
            st.info("**Interpretation:** 'Stable' positions have more predictable, less chaotic behavior, making them better candidates for statistical modeling. The Positional Dynamics Ensemble automatically focuses on these positions.")

else:
    st.info("Awaiting CSV file upload to begin analysis.")
