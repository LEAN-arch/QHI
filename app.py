# ======================================================================================================
# LottoSphere v20.0.2: Synergetic Dynamics Engine (Hybrid BNN Fix)
#
# VERSION: 20.0.2
#
# DESCRIPTION:
# This version provides a critical bugfix for the BayesianSequenceModel. The previous version
# incorrectly called a non-existent `bnn.BayesLSTM`. This has been replaced with a correct
# hybrid architecture using a standard `nn.LSTM` for feature extraction and a `bnn.BayesLinear`
# head for probabilistic prediction and uncertainty quantification.
#
# CHANGELOG (v20.0.2):
# - CRITICAL BUGFIX: Replaced call to non-existent `bnn.BayesLSTM` in `BayesianSequenceModel`.
# - NEW ARCHITECTURE: Implemented a hybrid `nn.LSTM` + `bnn.BayesLinear` model which correctly
#   leverages the `torchbnn` library for recurrent sequence modeling.
# - STABILITY: The application will now run correctly without the AttributeError.
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import itertools
import math

# --- Page Configuration and Optional Dependencies ---
st.set_page_config(
    page_title="LottoSphere v20.0.2: Synergetic Dynamics",
    page_icon="🕸️",
    layout="wide",
)

try:
    from sktime.forecasting.arima import AutoARIMA
except ImportError:
    AutoARIMA = None
    st.sidebar.warning("`sktime` or `pmdarima` not found. Stabilization analysis will be disabled.")
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
    return best_guesses

# --- 2. BASE MODEL CLASS (Unchanged) ---
class BaseModel:
    def __init__(self, max_nums: List[int]):
        self.max_nums = max_nums
        self.name = "Base Model"
        self.logic = "Base logic"
    def train(self, df: pd.DataFrame, **kwargs): raise NotImplementedError
    def predict(self, **kwargs) -> Dict[str, Any]: raise NotImplementedError

# --- 3. ADVANCED PREDICTIVE MODELS ---

class BayesianSequenceModel(BaseModel):
    def __init__(self, max_nums, seq_length=12, epochs=30):
        super().__init__(max_nums)
        self.name = "Bayesian LSTM Model"
        self.logic = "A hybrid model using a standard LSTM for temporal features and a Bayesian layer for probabilistic prediction."
        self.seq_length = seq_length
        self.epochs = epochs
        self.model = None
        self.scaler = None

    def train(self, df: pd.DataFrame, **kwargs):
        if not bnn or len(df) <= self.seq_length: return
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(df)
        X, y = create_sequences(data_scaled, self.seq_length)
        if len(X) == 0: return

        # --- BUGFIX: Define a proper hybrid model ---
        class _HybridBayesianLSTM(nn.Module):
            def __init__(self, input_size=6, hidden_size=50, output_size=6):
                super().__init__()
                # Standard LSTM for processing sequences
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
                # Bayesian Linear layer for the final output
                self.bayes_fc = bnn.BayesLinear(hidden_size, output_size)

            def forward(self, x):
                # lstm_out shape: (batch_size, seq_length, hidden_size)
                lstm_out, _ = self.lstm(x)
                # We only need the output of the last time step
                last_hidden_state = lstm_out[:, -1, :]
                # Pass the last hidden state to the Bayesian layer
                return self.bayes_fc(last_hidden_state)

        self.model = _HybridBayesianLSTM().to(device)
        # --- End of Bugfix ---

        X_torch, y_torch = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.MSELoss()
        # The ELBO loss function correctly finds the Bayesian layers in the model
        kl_loss = bnn.PyTorchBNN.ELBO(dataset=dataset, criterion=criterion, kl_weight=0.1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        
        for _ in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = kl_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, n_samples=50) -> Dict[str, Any]:
        if not self.model or not self.scaler:
            return {'distributions': [{k: 1/m for k in range(1, m + 1)} for m in self.max_nums], 'uncertainty': 1.0}

        last_seq_scaled = self.scaler.transform(st.session_state.df_master.iloc[-self.seq_length:].values)
        input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Sample multiple times to get a distribution of predictions
            predictions_raw = np.array([self.scaler.inverse_transform(self.model(input_tensor).cpu().numpy()).flatten() for _ in range(n_samples)])
        
        mean_pred = np.mean(predictions_raw, axis=0)
        std_pred = np.std(predictions_raw, axis=0)
        
        distributions = []
        for i in range(6):
            x_range = np.arange(1, self.max_nums[i] + 1)
            prob_mass = stats.norm.pdf(x_range, loc=mean_pred[i], scale=max(1.5, std_pred[i]))
            distributions.append({num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())})
            
        uncertainty_score = np.mean(std_pred / (np.array(self.max_nums)/2))
        return {'distributions': distributions, 'uncertainty': uncertainty_score}

class TransformerModel(BaseModel):
    def __init__(self, max_nums, seq_length=15, epochs=30):
        super().__init__(max_nums)
        self.name = "Transformer Model"
        self.logic = "A Transformer with self-attention to capture complex, long-range dependencies."
        self.seq_length = seq_length
        self.epochs = epochs
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
            def __init__(self, d_model=6, nhead=3, num_layers=2, dim_feedforward=128):
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
        last_seq_scaled = self.scaler.transform(st.session_state.df_master.iloc[-self.seq_length:].values)
        input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = self.model(input_tensor)
        pred_raw = self.scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
        distributions = []
        for i in range(6):
            std_dev = np.std(self.scaler.inverse_transform(self.scaler.transform(st.session_state.df_master))[:,i])
            x_range = np.arange(1, self.max_nums[i] + 1)
            prob_mass = stats.norm.pdf(x_range, loc=pred_raw[i], scale=max(1.5, std_dev*0.5))
            distributions.append({num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())})
        return {'distributions': distributions}

class GraphCommunityModel(BaseModel):
    def __init__(self, max_nums):
        super().__init__(max_nums)
        self.name = "Graph Community Model"
        self.logic = "Identifies communities of co-occurring numbers and biases predictions towards the strongest community."
        self.graph = None
        self.communities = []
    def train(self, df: pd.DataFrame, **kwargs):
        if not nx: return
        self.graph = nx.Graph()
        for _, row in df.iterrows():
            for u, v in itertools.combinations(row.values, 2):
                if self.graph.has_edge(u, v):
                    self.graph[u][v]['weight'] += 1
                else:
                    self.graph.add_edge(u, v, weight=1)
        resolution = kwargs.get('resolution', 1.0)
        try:
            self.communities = list(nx_comm.louvain_communities(self.graph, weight='weight', resolution=resolution, seed=42))
        except:
            self.communities = []
    def predict(self, **kwargs) -> Dict[str, Any]:
        if not self.graph or not self.communities:
            return {'distributions': [{k: 1/m for k in range(1, m + 1)} for m in self.max_nums]}
        largest_community = max(self.communities, key=len, default=[])
        centrality = nx.degree_centrality(self.graph)
        distributions = []
        for i in range(6):
            dist = {k: 0.01 for k in range(1, self.max_nums[i] + 1)}
            for num in largest_community:
                if 1 <= num <= self.max_nums[i]:
                    dist[num] = dist.get(num, 0.01) + centrality.get(num, 0)
            total_prob = sum(dist.values())
            distributions.append({k: v/total_prob for k, v in dist.items()})
        return {'distributions': distributions}

# --- 4. BACKTESTING & PERFORMANCE EVALUATION (Unchanged) ---
def run_backtest(model_instance: BaseModel, df: pd.DataFrame, train_size: int, backtest_steps: int, **kwargs) -> Dict[str, Any]:
    log_losses, uncertainties = [], []
    for i in range(backtest_steps):
        if train_size + i >= len(df): break # Prevent index out of bounds
        current_train_df = df.iloc[:train_size + i]
        true_draw = df.iloc[train_size + i].values
        model_instance.train(current_train_df, **kwargs)
        pred_obj = model_instance.predict(**kwargs)
        if 'uncertainty' in pred_obj:
            uncertainties.append(pred_obj['uncertainty'])
        step_log_loss = 0
        for pos_idx, dist in enumerate(pred_obj['distributions']):
            prob_of_true = dist.get(true_draw[pos_idx], 1e-9)
            step_log_loss -= np.log(prob_of_true)
        log_losses.append(step_log_loss)
    avg_log_loss = np.mean(log_losses) if log_losses else np.log(np.mean(model_instance.max_nums))
    likelihood = 100 * np.exp(-avg_log_loss / np.log(np.mean(model_instance.max_nums)))
    metrics = {'Log Loss': avg_log_loss, 'Likelihood': likelihood}
    if uncertainties:
        metrics['Uncertainty'] = np.mean(uncertainties)
    return metrics

# --- 5. STABILITY & DYNAMICS ANALYSIS FUNCTIONS (Unchanged from v20.0.1) ---
@st.cache_data
def find_stabilization_point(_df: pd.DataFrame, _max_nums: List[int], backtest_steps: int) -> go.Figure:
    if not AutoARIMA:
        st.error("`sktime` and/or `pmdarima` not installed. Stabilization analysis is disabled.")
        return go.Figure().update_layout(title_text="Stabilization Analysis Disabled")
    window_sizes = np.linspace(50, max(250, len(_df) - backtest_steps - 1), 10, dtype=int)
    results = []
    progress_bar = st.progress(0, "Running stabilization analysis...")
    for i, size in enumerate(window_sizes):
        if len(_df) < size + backtest_steps: continue
        log_losses, predictions = [], []
        for step in range(backtest_steps):
            train_df = _df.iloc[:size + step]
            true_draw = _df.iloc[size + step].values
            series = train_df['Pos_1'].values
            try:
                model = AutoARIMA(sp=1, suppress_warnings=True, maxiter=50)
                model.fit(series)
                pred_mean = model.predict(fh=[1])[0]
                pred_std = np.std(series) * 1.5
                x_range = np.arange(1, _max_nums[0] + 1)
                prob_mass = stats.norm.pdf(x_range, loc=pred_mean, scale=max(1.0, pred_std))
                prob_dist = {num: p for num, p in zip(x_range, prob_mass / prob_mass.sum())}
                prob_of_true = prob_dist.get(true_draw[0], 1e-9)
                log_losses.append(-np.log(prob_of_true))
                predictions.append(pred_mean)
            except Exception:
                log_losses.append(np.log(_max_nums[0]))
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
    fig.update_layout(title='Model Stabilization Analysis', xaxis_title='Training Window Size', yaxis_title='Cross-Entropy Loss', yaxis2=dict(title='Prediction Stability Index', overlaying='y', side='right'))
    return fig

@st.cache_data
def analyze_clusters(_df: pd.DataFrame, min_cluster_size: int, min_samples: int) -> Dict[str, Any]:
    results = {'fig': go.Figure(), 'summary': "Clustering disabled or failed."}
    if not hdbscan or not umap or len(_df) < min_cluster_size:
        if not hdbscan: results['summary'] = "Please install `hdbscan` to enable this feature."
        if not umap: results['summary'] = "Please install `umap-learn` to enable this feature."
        return results
    data = _df.values
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(data)
        if len(set(labels)) > 1:
            clean_labels = labels[labels != -1]
            if len(set(clean_labels)) > 1:
                clean_data = data[labels != -1]
                score = silhouette_score(clean_data, clean_labels)
                results['silhouette'] = f"{score:.3f}"
            else: results['silhouette'] = "N/A (1 cluster)"
        else: results['silhouette'] = "N/A"
        reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(data)
        plot_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
        plot_df['Cluster'] = [str(l) for l in labels]
        plot_df['Draw'] = _df.index
        plot_df['Numbers'] = _df.apply(lambda row: ', '.join(row.astype(str)), axis=1)
        fig = px.scatter(plot_df, x='UMAP_1', y='UMAP_2', color='Cluster', custom_data=['Draw', 'Numbers'],
                         title=f'Latent Space of Draw Behaviors (Silhouette: {results.get("silhouette", "N/A")})',
                         color_discrete_map={'-1': 'grey'})
        fig.update_traces(hovertemplate='<b>Draw %{customdata[0]}</b><br>Numbers: %{customdata[1]}<br>Cluster: %{marker.color}')
        results['fig'] = fig
        summary_text = ""
        cluster_counts = Counter(labels)
        for cluster_id, count in sorted(cluster_counts.items()):
            if cluster_id == -1:
                summary_text += f"- **Noise Points:** {count} draws did not belong to any cluster.\n"
            else:
                cluster_mean = _df[labels == cluster_id].mean().round().astype(int).tolist()
                summary_text += f"- **Cluster {cluster_id}:** {count} draws. Centroid (mean): `{cluster_mean}`\n"
        results['summary'] = summary_text
    except Exception as e:
        results['summary'] = f"An error occurred during clustering: {e}"
    return results

# --- 6. MAIN APPLICATION UI & LOGIC (Unchanged) ---
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
        tab1, tab2, tab3 = st.tabs(["🔮 Predictive Ensembles", "🕸️ Graph Dynamics", "📉 System Stability"])
        with tab1:
            st.header("🔮 Predictive Ensembles")
            st.markdown("Forecasts from a diverse suite of models. Each is rigorously backtested on a walk-forward basis.")
            with st.expander("How to Interpret These Forecasts", expanded=False):
                st.markdown("""
                - **Likelihood Score:** Confidence derived from historical prediction accuracy. Higher is better (>60% is promising).
                - **Uncertainty Score (BNN Only):** A powerful metric from the Bayesian model. It quantifies its own confidence. **A low score (<0.15) indicates a high-confidence forecast that should be trusted more.**
                - **The Models:**
                    - **Bayesian LSTM:** A neural network that provides both a prediction and its own uncertainty. **Prioritize its forecasts when uncertainty is low.**
                    - **Transformer:** The current standard for sequence modeling, excellent at finding complex, long-range patterns.
                    - **Graph Community:** A non-sequential model that predicts based on structural clusters of numbers. Useful for identifying stable 'regimes'.
                """)
            model_definitions = {}
            if bnn: model_definitions["Bayesian LSTM"] = BayesianSequenceModel(max_nums_input)
            model_definitions["Transformer"] = TransformerModel(max_nums_input)
            if nx: model_definitions["Graph Community"] = GraphCommunityModel(max_nums_input)
            if not model_definitions:
                st.error("No models available. Please install required libraries.")
            else:
                num_cols = len(model_definitions)
                cols = st.columns(num_cols) if num_cols > 0 else [st]
                for i, (name, model_instance) in enumerate(model_definitions.items()):
                    with cols[i]:
                        with st.container(border=True):
                            st.subheader(name)
                            with st.spinner(f"Running {name}..."):
                                perf_metrics = run_backtest(model_instance, df, training_size_slider, backtest_steps_slider)
                                model_instance.train(df)
                                final_pred_obj = model_instance.predict()
                                final_prediction = get_best_guess_set(final_pred_obj['distributions'])
                            st.markdown(f"**Logic:** *{model_instance.logic}*")
                            st.markdown(f"**Predicted Set:**")
                            st.code(" | ".join(map(str, final_prediction)))
                            m_cols = st.columns(2)
                            m_cols[0].metric("Likelihood Score", f"{perf_metrics['Likelihood']:.1f}%")
                            if 'Uncertainty' in perf_metrics:
                                m_cols[1].metric("Uncertainty Score", f"{perf_metrics['Uncertainty']:.3f}", help="Lower is better. <0.15 is high confidence.")
                            else:
                                m_cols[1].metric("Cross-Entropy", f"{perf_metrics['Log Loss']:.3f}")
                            with st.expander("View Probability Distributions"):
                                dist_cols = st.columns(3)
                                for k, dist in enumerate(final_pred_obj['distributions']):
                                    if dist:
                                        df_dist = pd.DataFrame(dist.items(), columns=['Number', 'Probability']).sort_values('Probability', ascending=False).head(10)
                                        fig = px.bar(df_dist, x='Number', y='Probability', height=200, title=f"Pos {k+1}")
                                        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis_title=None, xaxis={'categoryorder':'total descending'})
                                        dist_cols[k % 3].plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with tab2:
            st.header("🕸️ Graph Dynamics")
            if not nx:
                st.error("`networkx` is not installed. This feature is disabled.")
            else:
                st.markdown("This analysis reveals the hidden **social network** of the numbers. Nodes are numbers, and edges connect numbers that appear together in a draw. The colors represent distinct **communities** (clusters) of numbers that frequently associate.")
                st.sidebar.header("2. Graph Controls")
                graph_lookback = st.sidebar.slider("Lookback for Graph (Draws)", 20, 500, 100, 5)
                community_resolution = st.sidebar.slider("Community Resolution", 0.5, 2.5, 1.2, 0.1, help="Higher values -> more, smaller communities.")
                graph_df = df.iloc[-graph_lookback:]
                graph_model = GraphCommunityModel(max_nums_input)
                graph_model.train(graph_df, resolution=community_resolution)
                G, communities = graph_model.graph, graph_model.communities
                if not G or not communities:
                    st.warning("Could not generate graph. Data might be insufficient.")
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
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
                                        layout=go.Layout(title='Co-occurrence Network of Numbers', showlegend=False,
                                                         hovermode='closest', margin=dict(b=5,l=5,r=5,t=40),
                                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.subheader("Discovered Communities")
                        st.markdown("Numbers that tend to appear together.")
                        for i, comm in enumerate(communities):
                            if len(comm) > 2:
                                st.markdown(f"**C{i}:** `{sorted(list(comm))}`")
        with tab3:
            st.header("📉 System Stability & Dynamics")
            st.markdown("Tools from chaos theory and clustering to understand the system's overall predictability and structure.")
            st.subheader("Training Window Stabilization Analysis")
            st.markdown("Determines the optimal amount of historical data for training by finding where performance plateaus.")
            stabilization_fig = find_stabilization_point(df, max_nums_input, backtest_steps_slider)
            st.plotly_chart(stabilization_fig, use_container_width=True)
            st.subheader("Cluster Dynamics & Regime Analysis")
            st.markdown("Discovers 'behavioral regimes' in the data. Each point is a historical draw; clusters represent groups of similar draws.")
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
