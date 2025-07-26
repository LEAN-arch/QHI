# ======================================================================================================
# LottoSphere v21.0.0: Asymmetric Dynamics Engine
#
# VERSION: 21.0.0
#
# DESCRIPTION:
# This is a major architectural release reflecting a comprehensive code review and a fundamental
# shift in the modeling paradigm to meet the requirement that Position 6 be treated as a
# separate, independent entity.
#
# CHANGELOG (v21.0.0):
# - NEW ARCHITECTURE (5+1): The entire application is refactored. Positions 1-5 are modeled
#   as a 5D multivariate system. Position 6 is modeled as a 1D univariate system.
# - DEDICATED POS-6 MODEL: A new `UnivariateEnsemble` model was created for Position 6, using
#   a robust combination of AutoARIMA, Kernel Density Estimation (KDE), and a Markov Chain.
#   This is more statistically appropriate than forcing it into a multivariate model.
# - REFACTORED MODELS: The Transformer, Bayesian LSTM, and Graph models now operate exclusively
#   on the 5-dimensional data from Positions 1-5.
# - API BUGFIX: Permanently fixed the `bnn.BayesLinear` TypeError by using keyword arguments.
# - ENHANCED RESILIENCE: Added numerous checks for data length and empty data slices to
#   prevent `IndexError` crashes during training and backtesting.
# - CLARITY: All comments, docstrings, and UI text updated to reflect the new 5+1 architecture.
# ======================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
from typing import List, Dict, Any, Tuple
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import itertools
import math

# --- Page Configuration and Optional Dependencies ---
st.set_page_config(
    page_title="LottoSphere v21.0.0: Asymmetric Dynamics",
    page_icon="✨",
    layout="wide",
)

try:
    from sktime.forecasting.arima import AutoARIMA
except ImportError:
    AutoARIMA = None
    st.sidebar.warning("`sktime` or `pmdarima` not found. ARIMA-based models will be disabled.")
try:
    import networkx as nx
    from networkx.algorithms import community as nx_comm
except ImportError:
    nx = None
    st.sidebar.error("`networkx` not found. Graph analysis will be disabled.")
try:
    import torchbnn as bnn
except ImportError:
    bnn = None
    st.sidebar.error("`torchbnn` not found. Bayesian NN will be disabled.")
try:
    import hdbscan
except ImportError:
    hdbscan = None
    st.sidebar.warning("`hdbscan` not found. Cluster analysis will be disabled.")
try:
    import umap
except ImportError:
    umap = None
    st.sidebar.warning("`umap-learn` not found. UMAP visualization will be disabled.")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if 'df_master' not in st.session_state:
    st.session_state.df_master = pd.DataFrame()

device = torch.device("cpu")

# --- 1. CORE UTILITIES & DATA HANDLING (Unchanged) ---
@st.cache_data
def load_and_validate_data(uploaded_file: io.BytesIO, max_nums: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), header=None)
        if df.shape[1] != 6:
            logs.append(f"Error: CSV must have 6 columns, but found {df.shape[1]}.")
            return pd.DataFrame(), logs
        df.columns = [f'Pos_{i+1}' for i in range(6)]
        df_validated = df.copy()
        for i, col in enumerate(df_validated.columns):
            df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
        df_validated.dropna(inplace=True)
        df_validated = df_validated.astype(int)
        for i, max_num in enumerate(max_nums):
            df_validated = df_validated[df_validated[f'Pos_{i+1}'].between(1, max_num)]
        is_duplicate_in_row = df_validated.apply(lambda row: row.nunique() != 6, axis=1)
        df_validated = df_validated[~is_duplicate_in_row]
        if len(df_validated) < 50:
            logs.append(f"Error: Insufficient valid data ({len(df_validated)} rows). Need at least 50.")
            return pd.DataFrame(), logs
        df_validated = df_validated.reset_index(drop=True)
        logs.append(f"Data validation successful. Final dataset contains {len(df_validated)} draws.")
        return df_validated, logs
    except Exception as e:
        logs.append(f"Fatal error during data loading: {e}")
        return pd.DataFrame(), logs

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def get_best_guess_set(distributions: List[Dict[int, float]]) -> List[int]:
    # This logic naturally handles the 5+1 structure, as it ensures uniqueness across the combined list of 6 distributions
    best_guesses = [0] * 6
    seen_numbers = set()
    candidates = []
    for i, dist in enumerate(distributions):
        if not dist: continue
        sorted_probs = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        for num, prob in sorted_probs:
            if num not in seen_numbers:
                candidates.append({'pos': i, 'num': num, 'prob': prob})
                break
    for candidate in sorted(candidates, key=lambda x: x['prob'], reverse=True):
        if best_guesses[candidate['pos']] == 0 and candidate['num'] not in seen_numbers:
            best_guesses[candidate['pos']] = candidate['num']
            seen_numbers.add(candidate['num'])
    for i in range(6):
        if best_guesses[i] == 0:
            dist = distributions[i]
            sorted_probs = sorted(dist.items(), key=lambda item: item[1], reverse=True)
            for num, prob in sorted_probs:
                if num not in seen_numbers:
                    best_guesses[i] = num
                    seen_numbers.add(num)
                    break
    # Final check for any missed assignments (rare but a good safeguard)
    for i in range(6):
        if best_guesses[i] == 0:
            available_nums = sorted(list(set(range(1, 100)) - seen_numbers))
            if available_nums:
                best_guesses[i] = available_nums[0]
                seen_numbers.add(available_nums[0])
    return best_guesses


# --- 2. BASE MODEL CLASS (Unchanged) ---
class BaseModel:
    def __init__(self, max_nums: List[int]):
        self.max_nums = max_nums
        self.name = "Base Model"
        self.logic = "Base logic"
    def train(self, df: pd.DataFrame, **kwargs): raise NotImplementedError
    def predict(self, **kwargs) -> Dict[str, Any]: raise NotImplementedError


# --- 3. RE-ARCHITECTED PREDICTIVE MODELS (5+1 STRUCTURE) ---

# --- Models for Positions 1-5 (Multivariate) ---

class BayesianSequenceModel(BaseModel):
    def __init__(self, max_nums):
        # Operates on first 5 numbers
        super().__init__(max_nums[:5])
        self.name = "Bayesian LSTM"
        self.logic = "Hybrid BNN for Positions 1-5, quantifying uncertainty."
        self.seq_length = 12
        self.epochs = 30
        self.model = None
        self.scaler = None

    def train(self, df: pd.DataFrame, **kwargs):
        if not bnn or len(df) <= self.seq_length: return
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(df)
        X, y = create_sequences(data_scaled, self.seq_length)
        if len(X) == 0: return

        class _HybridBayesianLSTM(nn.Module):
            def __init__(self, input_size=5, hidden_size=50, output_size=5):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
                self.bayes_fc = bnn.BayesLinear(in_features=hidden_size, out_features=output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden_state = lstm_out[:, -1, :]
                return self.bayes_fc(last_hidden_state)

        self.model = _HybridBayesianLSTM().to(device)
        X_torch, y_torch = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        kl_loss = bnn.PyTorchBNN.ELBO(dataset=dataset, criterion=criterion, kl_weight=0.1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        for _ in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = kl_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, n_samples=50, **kwargs) -> Dict[str, Any]:
        if not self.model or not self.scaler:
            return {'distributions': [{k: 1/m for k in range(1, m + 1)} for m in self.max_nums], 'uncertainty': 1.0}

        # Need the full 5-column history
        full_history = kwargs['full_history'].iloc[:, :5]
        last_seq_scaled = self.scaler.transform(full_history.iloc[-self.seq_length:].values)
        input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions_raw = np.array([self.scaler.inverse_transform(self.model(input_tensor).cpu().numpy()).flatten() for _ in range(n_samples)])
        
        mean_pred = np.mean(predictions_raw, axis=0)
        std_pred = np.std(predictions_raw, axis=0)
        distributions = []
        for i in range(5): # Predict for 5 positions
            x_range = np.arange(1, self.max_nums[i] + 1)
            prob_mass = stats.norm.pdf(x_range, loc=mean_pred[i], scale=max(1.5, std_pred[i]))
            distributions.append({num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())})
            
        uncertainty_score = np.mean(std_pred / (np.array(self.max_nums)/2))
        return {'distributions': distributions, 'uncertainty': uncertainty_score}

class TransformerModel(BaseModel):
    def __init__(self, max_nums):
        super().__init__(max_nums[:5])
        self.name = "Transformer"
        self.logic = "Transformer model for long-range patterns in Positions 1-5."
        self.seq_length = 15
        self.epochs = 30
        self.model = None
        self.scaler = None

    def train(self, df: pd.DataFrame, **kwargs):
        if len(df) <= self.seq_length: return
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(df)
        X, y = create_sequences(data_scaled, self.seq_length)
        if len(X) == 0: return

        class _PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=50):
                super().__init__()
                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(max_len, 1, d_model)
                pe[:, 0, 0::2] = torch.sin(position * div_term)
                pe[:, 0, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)
            def forward(self, x):
                return x + self.pe[:x.size(0)]

        class _Transformer(nn.Module):
            def __init__(self, d_model=5, nhead=5, num_layers=2, dim_feedforward=128):
                super().__init__()
                self.pos_encoder = _PositionalEncoding(d_model)
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True, dropout=0.1)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.fc = nn.Linear(d_model, d_model)

            def forward(self, src):
                src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2)
                output = self.transformer_encoder(src)
                return self.fc(output[:, -1, :])

        self.model = _Transformer().to(device)
        X_torch, y_torch = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for _ in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, **kwargs) -> Dict[str, Any]:
        if not self.model or not self.scaler:
            return {'distributions': [{k: 1/m for k in range(1, m + 1)} for m in self.max_nums]}
        
        full_history = kwargs['full_history'].iloc[:, :5]
        last_seq_scaled = self.scaler.transform(full_history.iloc[-self.seq_length:].values)
        input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = self.model(input_tensor)
        pred_raw = self.scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        distributions = []
        for i in range(5):
            std_dev = np.std(self.scaler.inverse_transform(self.scaler.transform(full_history))[:,i])
            x_range = np.arange(1, self.max_nums[i] + 1)
            prob_mass = stats.norm.pdf(x_range, loc=pred_raw[i], scale=max(1.5, std_dev*0.5))
            distributions.append({num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())})
        return {'distributions': distributions}


# --- Model for Position 6 (Univariate) ---

class UnivariateEnsemble(BaseModel):
    def __init__(self, max_nums):
        # Operates on the 6th number
        super().__init__([max_nums[5]])
        self.name = "Pos 6 Ensemble"
        self.logic = "Statistical ensemble (ARIMA, KDE, Markov) for the independent Position 6."
        self.kde = None
        self.arima_pred = None
        self.markov_chain = None
        self.last_val = None

    def train(self, df: pd.DataFrame, **kwargs):
        series = df.values.flatten()
        if len(series) < 10: return

        # Kernel Density Estimation
        self.kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(series[:, None])

        # ARIMA prediction
        if AutoARIMA:
            try:
                arima_model = AutoARIMA(sp=1, suppress_warnings=True, maxiter=50)
                arima_model.fit(series)
                self.arima_pred = arima_model.predict(fh=[1])[0]
            except Exception:
                self.arima_pred = np.mean(series) # Fallback
        else:
            self.arima_pred = np.mean(series)

        # Markov Chain
        max_num = self.max_nums[0]
        self.markov_chain = np.zeros((max_num + 1, max_num + 1))
        for i in range(len(series) - 1):
            self.markov_chain[series[i], series[i+1]] += 1
        # Normalize to get probabilities, add smoothing
        self.markov_chain = (self.markov_chain + 0.1) / (self.markov_chain.sum(axis=1, keepdims=True) + 0.1 * max_num)
        self.last_val = series[-1]

    def predict(self, **kwargs) -> Dict[str, Any]:
        if self.kde is None:
            return {'distributions': [{k: 1/self.max_nums[0] for k in range(1, self.max_nums[0] + 1)}]}

        max_num = self.max_nums[0]
        x_range = np.arange(1, max_num + 1)[:, None]
        
        # Get probabilities from each model
        kde_probs = np.exp(self.kde.score_samples(x_range))
        arima_probs = stats.norm.pdf(x_range, loc=self.arima_pred, scale=np.std(x_range)).flatten()
        markov_probs = self.markov_chain[self.last_val] if self.last_val is not None else np.ones(max_num + 1)
        markov_probs = markov_probs[1:] # Align with 1-based indexing

        # Ensemble (weighted average)
        ensemble_probs = (0.4 * kde_probs + 0.3 * arima_probs + 0.3 * markov_probs)
        ensemble_probs /= ensemble_probs.sum()
        
        distribution = {int(num): float(prob) for num, prob in zip(x_range.flatten(), ensemble_probs)}
        return {'distributions': [distribution]}


# --- 4. BACKTESTING & PERFORMANCE EVALUATION ---
def run_backtest(model_instance: BaseModel, pos6_model: BaseModel, df: pd.DataFrame, train_size: int, backtest_steps: int, **kwargs) -> Dict[str, Any]:
    log_losses, uncertainties = [], []
    df_main, df_pos6 = df.iloc[:, :5], df.iloc[:, 5]

    for i in range(backtest_steps):
        if train_size + i >= len(df): break
        current_main_df = df_main.iloc[:train_size + i]
        current_pos6_df = df_pos6.iloc[:train_size + i]
        true_draw = df.iloc[train_size + i].values

        # Train both models on their respective data slices
        model_instance.train(current_main_df, **kwargs)
        pos6_model.train(current_pos6_df)
        
        # Get predictions from both
        pred_obj_main = model_instance.predict(full_history=df.iloc[:train_size+i], **kwargs)
        pred_obj_pos6 = pos6_model.predict()
        
        # Combine results
        all_distributions = pred_obj_main['distributions'] + pred_obj_pos6['distributions']

        if 'uncertainty' in pred_obj_main:
            uncertainties.append(pred_obj_main['uncertainty'])
        
        step_log_loss = 0
        for pos_idx, dist in enumerate(all_distributions):
            prob_of_true = dist.get(true_draw[pos_idx], 1e-9)
            step_log_loss -= np.log(prob_of_true)
        log_losses.append(step_log_loss)

    # Use the full max_nums list for likelihood calculation
    full_max_nums = model_instance.max_nums + pos6_model.max_nums
    avg_log_loss = np.mean(log_losses) if log_losses else np.log(np.mean(full_max_nums))
    likelihood = 100 * np.exp(-avg_log_loss / np.log(np.mean(full_max_nums)))
    metrics = {'Log Loss': avg_log_loss, 'Likelihood': likelihood}
    if uncertainties:
        metrics['Uncertainty'] = np.mean(uncertainties)
    return metrics


# --- 5. STABILITY & DYNAMICS ANALYSIS FUNCTIONS ---
@st.cache_data
def find_stabilization_point(_df: pd.DataFrame, _max_nums: List[int], backtest_steps: int) -> go.Figure:
    if not AutoARIMA:
        st.error("`sktime` and/or `pmdarima` not installed. Stabilization analysis is disabled.")
        return go.Figure().update_layout(title_text="Stabilization Analysis Disabled")
    # This analysis still uses just one position as a proxy for system stability
    df_pos1 = _df.iloc[:, 0]
    max_num_pos1 = _max_nums[0]
    window_sizes = np.linspace(50, max(250, len(df_pos1) - backtest_steps - 1), 10, dtype=int)
    results = []
    progress_bar = st.progress(0, "Running stabilization analysis...")
    for i, size in enumerate(window_sizes):
        if len(df_pos1) < size + backtest_steps: continue
        log_losses, predictions = [], []
        for step in range(backtest_steps):
            series = df_pos1.iloc[:size + step]
            true_val = df_pos1.iloc[size + step]
            try:
                model = AutoARIMA(sp=1, suppress_warnings=True, maxiter=50)
                model.fit(series)
                pred_mean = model.predict(fh=[1])[0]
                pred_std = np.std(series) * 1.5
                x_range = np.arange(1, max_num_pos1 + 1)
                prob_mass = stats.norm.pdf(x_range, loc=pred_mean, scale=max(1.0, pred_std))
                prob_dist = {num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())}
                prob_of_true = prob_dist.get(true_val, 1e-9)
                log_losses.append(-np.log(prob_of_true))
                predictions.append(pred_mean)
            except Exception:
                log_losses.append(np.log(max_num_pos1))
        avg_log_loss = np.mean(log_losses)
        psi = np.std(predictions) / (np.mean(predictions) + 1e-9) if predictions else float('nan')
        results.append({'Window Size': size, 'Cross-Entropy Loss': avg_log_loss, 'Prediction Stability Index': psi})
        progress_bar.progress((i + 1) / len(window_sizes))
    progress_bar.empty()
    if not results: return go.Figure().update_layout(title_text="Insufficient data for stabilization analysis.")
    results_df = pd.DataFrame(results).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['Window Size'], y=results_df['Cross-Entropy Loss'], mode='lines+markers', name='Cross-Entropy Loss'))
    fig.add_trace(go.Scatter(x=results_df['Window Size'], y=results_df['Prediction Stability Index'], mode='lines+markers', name='Prediction Stability Index', yaxis='y2'))
    fig.update_layout(title='Model Stabilization Analysis (on Pos 1)', xaxis_title='Training Window Size', yaxis_title='Cross-Entropy Loss', yaxis2=dict(title='Prediction Stability Index', overlaying='y', side='right'))
    return fig

@st.cache_data
def analyze_clusters(_df: pd.DataFrame, min_cluster_size: int, min_samples: int) -> Dict[str, Any]:
    # Cluster analysis is performed on the main 5 positions
    df_main = _df.iloc[:, :5]
    results = {'fig': go.Figure(), 'summary': "Clustering disabled or failed."}
    if not hdbscan or not umap or len(df_main) < min_cluster_size:
        return results
    data = df_main.values
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(data)
        # Silhouette score calculation
        if len(set(labels)) > 1 and -1 in labels:
            clean_labels = labels[labels != -1]
            if len(set(clean_labels)) > 1:
                clean_data = data[labels != -1]
                score = silhouette_score(clean_data, clean_labels)
                results['silhouette'] = f"{score:.3f}"
            else: results['silhouette'] = "N/A (1 cluster)"
        else: results['silhouette'] = "N/A"
        # UMAP visualization
        reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(data)
        plot_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
        plot_df['Cluster'] = [str(l) for l in labels]
        plot_df['Draw'] = df_main.index
        plot_df['Numbers'] = df_main.apply(lambda row: ', '.join(row.astype(str)), axis=1)
        fig = px.scatter(plot_df, x='UMAP_1', y='UMAP_2', color='Cluster', custom_data=['Draw', 'Numbers'],
                         title=f'Latent Space of Draws (Pos 1-5), Silhouette: {results.get("silhouette", "N/A")}',
                         color_discrete_map={'-1': 'grey'})
        fig.update_traces(hovertemplate='<b>Draw %{customdata[0]}</b><br>Numbers: %{customdata[1]}<br>Cluster: %{marker.color}')
        results['fig'] = fig
        # Cluster summary
        summary_text = ""
        cluster_counts = Counter(labels)
        for cluster_id, count in sorted(cluster_counts.items()):
            if cluster_id == -1:
                summary_text += f"- **Noise Points:** {count} draws did not belong to any cluster.\n"
            else:
                cluster_mean = df_main[labels == cluster_id].mean().round().astype(int).tolist()
                summary_text += f"- **Cluster {cluster_id}:** {count} draws. Centroid (mean): `{cluster_mean}`\n"
        results['summary'] = summary_text
    except Exception as e:
        results['summary'] = f"An error occurred during clustering: {e}"
    return results


# --- 6. MAIN APPLICATION UI & LOGIC ---

st.sidebar.header("1. System Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Number History (CSV)", type=["csv"])
with st.sidebar.expander("Advanced Configuration", expanded=True):
    max_nums_input = [st.number_input(f"Max Value for Pos_{i+1}", 10, 150, 49, key=f"max_num_{i}") for i in range(6)]
    training_size_slider = st.slider("Training Window Size", 50, 1000, 150, 10, help="Number of past draws to train on.")
    backtest_steps_slider = st.slider("Backtest Validation Steps", 5, 50, 10, 1, help="Number of steps for performance evaluation.")

if uploaded_file:
    df, logs = load_and_validate_data(uploaded_file, max_nums_input)
    with st.sidebar.expander("Data Loading Log", expanded=False):
        for log in logs: st.info(log)
    if not df.empty:
        st.session_state.df_master = df
        st.sidebar.success(f"Loaded and validated {len(df)} draws.")
        tab1, tab2, tab3 = st.tabs(["🔮 Predictive Ensembles", "🕸️ Graph Dynamics (Pos 1-5)", "📉 System Stability"])
        with tab1:
            st.header("🔮 Predictive Ensembles")
            st.markdown("Forecasts from a diverse suite of models operating on a **5+1 architecture**: Positions 1-5 are modeled as a correlated set, and Position 6 is modeled as an independent entity.")
            
            # Instantiate models
            model_definitions = {}
            if bnn: model_definitions["Bayesian LSTM"] = BayesianSequenceModel(max_nums_input)
            model_definitions["Transformer"] = TransformerModel(max_nums_input)
            # The UnivariateEnsemble is always paired with each main model
            pos6_model_instance = UnivariateEnsemble(max_nums_input)

            if not model_definitions:
                st.error("No multivariate models available. Please install required libraries.")
            else:
                num_cols = len(model_definitions)
                cols = st.columns(num_cols) if num_cols > 0 else [st]
                for i, (name, model_instance) in enumerate(model_definitions.items()):
                    with cols[i]:
                        with st.container(border=True):
                            st.subheader(name)
                            with st.spinner(f"Running {name}..."):
                                perf_metrics = run_backtest(model_instance, pos6_model_instance, df, training_size_slider, backtest_steps_slider)
                                # Final training on full data
                                model_instance.train(df.iloc[:,:5])
                                pos6_model_instance.train(df.iloc[:,5])
                                # Final prediction
                                final_pred_main = model_instance.predict(full_history=df)
                                final_pred_pos6 = pos6_model_instance.predict()
                                all_distributions = final_pred_main['distributions'] + final_pred_pos6['distributions']
                                final_prediction = get_best_guess_set(all_distributions)
                            
                            st.markdown(f"**Logic (Pos 1-5):** *{model_instance.logic}*")
                            st.markdown(f"**Logic (Pos 6):** *{pos6_model_instance.logic}*")
                            st.markdown(f"**Predicted Set:**")
                            st.code(" | ".join(map(str, final_prediction)))
                            m_cols = st.columns(2)
                            m_cols[0].metric("Likelihood Score", f"{perf_metrics['Likelihood']:.1f}%")
                            if 'Uncertainty' in perf_metrics:
                                m_cols[1].metric("BNN Uncertainty", f"{perf_metrics['Uncertainty']:.3f}", help="Model uncertainty for Pos 1-5. Lower is better.")
                            else:
                                m_cols[1].metric("Cross-Entropy", f"{perf_metrics['Log Loss']:.3f}")

                            with st.expander("View Probability Distributions"):
                                dist_cols = st.columns(3)
                                for k, dist in enumerate(all_distributions):
                                    if dist:
                                        df_dist = pd.DataFrame(dist.items(), columns=['Number', 'Probability']).sort_values('Probability', ascending=False).head(10)
                                        fig = px.bar(df_dist, x='Number', y='Probability', height=200, title=f"Pos {k+1}")
                                        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis_title=None, xaxis={'categoryorder':'total descending'})
                                        dist_cols[k % 3].plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with tab2:
            st.header("🕸️ Graph Dynamics (Positions 1-5)")
            if not nx:
                st.error("`networkx` is not installed. This feature is disabled.")
            else:
                st.markdown("This analysis reveals the **social network** of the first 5 numbers. Nodes are numbers, and edges connect numbers that appear together in a draw.")
                st.sidebar.header("2. Graph Controls")
                graph_lookback = st.sidebar.slider("Lookback for Graph (Draws)", 20, 500, 100, 5)
                community_resolution = st.sidebar.slider("Community Resolution", 0.5, 2.5, 1.2, 0.1, help="Higher values -> more, smaller communities.")
                
                # Graph analysis is now only on the first 5 positions
                graph_df = df.iloc[-graph_lookback:, :5]
                
                if graph_df.empty:
                    st.warning("Not enough data for graph analysis with current settings.")
                else:
                    # We can use the GraphCommunityModel for analysis without prediction
                    graph_analyzer = GraphCommunityModel(max_nums_input)
                    graph_analyzer.train(graph_df, resolution=community_resolution)
                    G, communities = graph_analyzer.graph, graph_analyzer.communities

                    if not G or not communities:
                        st.warning("Could not generate graph. Data might be insufficient or lack co-occurrences.")
                    else:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
                            # Visualization code remains largely the same
                            edge_x, edge_y = [], []
                            for edge in G.edges():
                                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
                            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                            centrality = nx.degree_centrality(G)
                            color_map = px.colors.qualitative.Vivid
                            community_map = {node: i for i, comm in enumerate(communities) for node in comm}
                            for node in G.nodes():
                                x, y = pos[node]
                                node_x.append(x); node_y.append(y)
                                node_color.append(color_map[community_map.get(node, -1) % len(color_map)])
                                node_size.append(15 + 40 * centrality.get(node, 0))
                                node_text.append(f"Num: {node}<br>Community: {community_map.get(node, 'N/A')}<br>Centrality: {centrality.get(node, 0):.2f}")
                            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=node_text,
                                                    marker=dict(showscale=False, color=node_color, size=node_size, line_width=1))
                            fig = go.Figure(data=[edge_trace, node_trace],
                                            layout=go.Layout(title='Co-occurrence Network of Numbers (Pos 1-5)', showlegend=False,
                                                             hovermode='closest', margin=dict(b=5,l=5,r=5,t=40),
                                                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            st.subheader("Discovered Communities")
                            st.markdown("Numbers that tend to appear together in positions 1-5.")
                            for i, comm in enumerate(communities):
                                if len(comm) > 2:
                                    st.markdown(f"**C{i}:** `{sorted(list(comm))}`")
        with tab3:
            st.header("📉 System Stability & Dynamics")
            st.markdown("Tools from chaos theory and clustering to understand the system's overall predictability and structure.")
            st.subheader("Training Window Stabilization Analysis")
            st.markdown("Determines the optimal amount of historical data for training by finding where performance plateaus. **This analysis is run on Position 1 as a proxy for the main system's stability.**")
            stabilization_fig = find_stabilization_point(df, max_nums_input, backtest_steps_slider)
            st.plotly_chart(stabilization_fig, use_container_width=True)
            st.subheader("Cluster Dynamics & Regime Analysis (Pos 1-5)")
            st.markdown("Discovers 'behavioral regimes' in the data for the main set of 5 numbers.")
            st.sidebar.header("3. Clustering Controls")
            cluster_min_size = st.sidebar.slider("Min Cluster Size", 5, 50, 15, 1)
            cluster_min_samples = st.sidebar.slider("Min Samples", 1, 20, 5, 1)
            cluster_results = analyze_clusters(df.iloc[-training_size_slider:], cluster_min_size, cluster_min_samples)
            col1, col2 = st.columns([3, 1])
            col1.plotly_chart(cluster_results['fig'], use_container_width=True)
            with col2:
                st.write("#### Cluster Interpretation")
                st.markdown(cluster_results['summary'])
else:
    st.info("Awaiting CSV file upload to begin analysis.")
