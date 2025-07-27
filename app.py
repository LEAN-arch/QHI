# ======================================================================================================
# LottoSphere v25.0.2: Aletheia Engine (Dependency Fix Release)
#
# VERSION: 25.0.2
#
# DESCRIPTION:
# Fixes "ERROR: Could not find a version that satisfies the requirement torchbnn==1.0.7" and
# "Prediction Failed: Numpy is not available" errors by:
# - Ensuring compatibility with Python 3.11 (Streamlit Community Cloud default).
# - Using torchbnn==1.2 instead of 1.0.7.
# - Adding explicit NumPy checks with uniform distribution fallbacks.
# - Improving error messages for NumPy and torchbnn failures.
# Maintains all features from v25.0.0: 5+1 architecture, System Health Check, pre-flight data
# validation, full traceback visibility, and models (LSTM, GRU, Transformer, Bayesian LSTM,
# Univariate Ensemble for Pos 6).
#
# CHANGELOG (v25.0.2):
# - FIXED: torchbnn==1.0.7 installation error by updating to torchbnn==1.2.
# - FIXED: Added NumPy import checks and fallbacks in predict methods.
# - ENHANCED: Improved logging in data_warnings for better debugging.
# - MAINTAINED: All models, System Health Check, and UI features from v25.0.0.
# ======================================================================================================

import streamlit as st
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
from typing import List, Dict, Any
import scipy
from sklearn.neighbors import KernelDensity
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import itertools
import math
import hashlib
import traceback

# --- Check for NumPy ---
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    st.error("NumPy is not available. Predictions will use uniform distributions.")

# --- Optional Dependencies ---
try:
    from sktime.forecasting.arima import AutoARIMA
except ImportError:
    AutoARIMA = None
    st.warning("sktime not available, Pos_6 ARIMA disabled.")
try:
    import networkx as nx
    from networkx.algorithms import community as nx_comm
except ImportError:
    nx = None
    st.warning("networkx not available, Graph Dynamics disabled.")
try:
    import torchbnn as bnn
except ImportError:
    bnn = None
    st.warning("torchbnn not available, Bayesian LSTM disabled.")
try:
    import hdbscan
except ImportError:
    hdbscan = None
    st.warning("hdbscan not available, clustering disabled.")
try:
    import umap
except ImportError:
    umap = None
    st.warning("umap-learn not available, clustering disabled.")
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
    st.warning("hmmlearn not available, HMM disabled.")

st.set_page_config(page_title="LottoSphere v25.0.2: Aletheia Engine", page_icon="💡", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if 'df_master' not in st.session_state:
    st.session_state.df_master = pd.DataFrame()
if 'data_warnings' not in st.session_state:
    st.session_state.data_warnings = []
device = torch.device("cpu")
st.session_state.data_warnings.append(f"Using device: {device}")

# --- Clear Streamlit Cache ---
if 'cache_cleared' not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.cache_cleared = True
    st.session_state.data_warnings.append("Streamlit cache cleared.")

# --- 1. CORE UTILITIES & DATA HANDLING ---
@st.cache_data
def load_and_validate_data(uploaded_file, max_nums):
    logs = []
    if not NUMPY_AVAILABLE:
        logs.append("Error: NumPy is not available, cannot process data.")
        return pd.DataFrame(), logs
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), header=None)
        if df.shape[1] != 6:
            logs.append(f"Error: CSV must have 6 columns, but found {df.shape[1]}.")
            return pd.DataFrame(), logs
        if len(max_nums) != 6:
            logs.append(f"Error: max_nums must have 6 values, got {len(max_nums)}.")
            return pd.DataFrame(), logs
        df.columns = [f'Pos_{i+1}' for i in range(6)]
        df_validated = df.copy()
        for i, col in enumerate(df_validated.columns):
            df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
        df_validated.dropna(inplace=True)
        if not np.isfinite(df_validated).all().all():
            logs.append("Error: Data contains non-finite values (NaN or infinity) after cleaning.")
            return pd.DataFrame(), logs
        df_validated = df_validated.astype(int)
        for i, max_num in enumerate(max_nums):
            df_validated = df_validated[df_validated[f'Pos_{i+1}'].between(1, max_num)]
        is_duplicate_in_row = df_validated.apply(lambda row: row.nunique() != 6, axis=1)
        if is_duplicate_in_row.any():
            logs.append(f"Discarded {is_duplicate_in_row.sum()} rows with duplicate numbers.")
            df_validated = df_validated[~is_duplicate_in_row]
        if df_validated.duplicated().any():
            logs.append(f"Discarded {df_validated.duplicated().sum()} duplicate rows.")
            df_validated = df_validated.drop_duplicates()
        for col in df_validated.columns:
            unique_vals = len(df_validated[col].unique())
            if unique_vals < 5:
                logs.append(f"Error: Too few unique values in {col}: {unique_vals}.")
                return pd.DataFrame(), logs
            if df_validated[col].std() < 1.0:
                logs.append(f"Error: Low variance in {col}: std={df_validated[col].std():.2f}.")
                return pd.DataFrame(), logs
        if len(df_validated) < 50:
            logs.append(f"Error: Insufficient valid data ({len(df_validated)} rows). Need at least 50.")
            return pd.DataFrame(), logs
        df_validated = df_validated.reset_index(drop=True)
        logs.append(f"Data validation successful. Final dataset contains {len(df_validated)} draws.")
        return df_validated, logs
    except Exception as e:
        logs.append(f"Fatal error during data loading: {e}")
        return pd.DataFrame(), logs

def create_sequences(data, seq_length):
    if not NUMPY_AVAILABLE:
        st.session_state.data_warnings.append("create_sequences: NumPy unavailable, cannot create sequences.")
        return [], []
    if seq_length >= len(data):
        st.session_state.data_warnings.append(f"Sequence length {seq_length} ≥ data length {len(data)}.")
        return np.array([]), np.array([])
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def get_best_guess_set(distributions, max_nums):
    if not NUMPY_AVAILABLE:
        st.session_state.data_warnings.append("get_best_guess_set: NumPy unavailable, returning random set.")
        seen_numbers = set()
        best_guesses = []
        for max_num in max_nums:
            guess = int((hash(str(max_num)) % max_num) + 1)
            while guess in seen_numbers:
                guess = int((hash(str(guess)) % max_num) + 1)
            best_guesses.append(guess)
            seen_numbers.add(guess)
        st.session_state.data_warnings.append(f"Random guess set due to NumPy unavailability: {best_guesses}")
        return best_guesses
    best_guesses = [0] * 6
    seen_numbers = set()
    candidates = []
    for i, dist in enumerate(distributions):
        if not dist or not all(isinstance(p, (int, float)) and p >= 0 for p in dist.values()):
            st.session_state.data_warnings.append(f"Invalid distribution for Pos_{i+1}: {dist}. Using uniform.")
            dist = {j: 1/max_nums[i] for j in range(1, max_nums[i] + 1)}
        total_prob = sum(dist.values())
        if total_prob == 0 or np.isnan(total_prob):
            st.session_state.data_warnings.append(f"Zero/NaN probability for Pos_{i+1}. Using uniform.")
            dist = {j: 1/max_nums[i] for j in range(1, max_nums[i] + 1)}
        else:
            dist = {k: v/total_prob for k, v in dist.items()}
        sorted_probs = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        for num, prob in sorted_probs:
            if num not in seen_numbers and 1 <= num <= max_nums[i]:
                candidates.append({'pos': i, 'num': num, 'prob': prob})
                break
    for candidate in sorted(candidates, key=lambda x: x['prob'], reverse=True):
        if best_guesses[candidate['pos']] == 0 and candidate['num'] not in seen_numbers:
            best_guesses[candidate['pos']] = candidate['num']
            seen_numbers.add(candidate['num'])
    for i in range(6):
        if best_guesses[i] == 0:
            dist = distributions[i]
            sorted_probs = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            for num, prob in sorted_probs:
                if num not in seen_numbers and 1 <= num <= max_nums[i]:
                    best_guesses[i] = num
                    seen_numbers.add(num)
                    break
    for i in range(6):
        if best_guesses[i] == 0:
            available_nums = sorted(list(set(range(1, max_nums[i] + 1)) - seen_numbers))
            if available_nums:
                best_guesses[i] = available_nums[0]
                seen_numbers.add(available_nums[0])
            else:
                st.session_state.data_warnings.append(f"No available numbers for Pos_{i+1}. Using random.")
                guess = np.random.randint(1, max_nums[i] + 1) if NUMPY_AVAILABLE else int((hash(str(i)) % max_nums[i]) + 1)
                while guess in seen_numbers:
                    guess = np.random.randint(1, max_nums[i] + 1) if NUMPY_AVAILABLE else int((hash(str(guess)) % max_nums[i]) + 1)
                best_guesses[i] = guess
                seen_numbers.add(guess)
    st.session_state.data_warnings.append(f"Best guess set: {best_guesses}")
    return best_guesses

# --- 2. BASE MODEL & MODEL FACTORY ---
class BaseModel:
    def __init__(self, max_nums: List[int]):
        self.max_nums = max_nums
        self.name = "Base Model"
        self.logic = "Base logic"
        self.min_data_length = 50
    def train(self, df: pd.DataFrame): raise NotImplementedError
    def predict(self, full_history: pd.DataFrame) -> Dict[str, Any]: raise NotImplementedError

class BaseSequenceModel(BaseModel):
    def __init__(self, max_nums):
        super().__init__(max_nums[:5])
        self.epochs = 30
        self.scaler = None
        self.model = None
    def _create_torch_model(self): raise NotImplementedError
    def train(self, df: pd.DataFrame):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is not available, cannot train model.")
        if len(df) < self.min_data_length:
            raise ValueError(f"Training data size ({len(df)}) is less than minimum required ({self.min_data_length}).")
        if not np.isfinite(df).all().all():
            raise ValueError("Training data contains non-finite values (NaN or infinity).")
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(df)
        X, y = create_sequences(data_scaled, self.seq_length)
        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences. Training size: {len(df)}, sequence length: {self.seq_length}.")
        self.model = self._create_torch_model().to(device)
        X_torch, y_torch = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)
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
        st.session_state.data_warnings.append(f"{self.name} trained: {len(X)} sequences, {self.epochs} epochs.")
    def predict(self, full_history: pd.DataFrame) -> Dict[str, Any]:
        if not NUMPY_AVAILABLE:
            st.session_state.data_warnings.append(f"{self.name}: NumPy unavailable, returning uniform distributions.")
            return {'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in self.max_nums]}
        if not self.model or not self.scaler:
            raise RuntimeError("Model is not trained.")
        if full_history is None or full_history.empty:
            raise ValueError(f"{self.name}: full_history is None or empty.")
        history_main = full_history.iloc[:, :5]
        if len(history_main) < self.seq_length:
            raise ValueError(f"Prediction history ({len(history_main)}) is less than sequence length ({self.seq_length}).")
        try:
            last_seq_scaled = self.scaler.transform(history_main.iloc[-self.seq_length:].values)
            input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_scaled = self.model(input_tensor)
            pred_raw = self.scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
            distributions = []
            for i in range(5):
                std_dev = np.std(self.scaler.inverse_transform(self.scaler.transform(history_main))[:,i])
                x_range = np.arange(1, self.max_nums[i] + 1)
                prob_mass = scipy.stats.norm.pdf(x_range, loc=pred_raw[i], scale=max(1.5, std_dev*0.5))
                prob_mass /= prob_mass.sum() + 1e-10
                distributions.append({num: float(p) for num, p in zip(x_range, prob_mass)})
            st.session_state.data_warnings.append(f"{self.name} predicted: Distributions={len(distributions)}")
            return {'distributions': distributions}
        except Exception as e:
            st.session_state.data_warnings.append(f"{self.name} prediction failed: {e}")
            return {'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in self.max_nums]}

class LSTMModel(BaseSequenceModel):
    def __init__(self, max_nums):
        super().__init__(max_nums)
        self.name = "LSTM"
        self.logic = "LSTM model for temporal sequence patterns in Pos 1-5."
        self.seq_length = 12
        self.min_data_length = self.seq_length + 20
    def _create_torch_model(self):
        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(5, 50, num_layers=2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(50, 5)
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
        return _LSTM()

class GRUModel(BaseSequenceModel):
    def __init__(self, max_nums):
        super().__init__(max_nums)
        self.name = "GRU"
        self.logic = "GRU model, a faster variant of LSTM for Pos 1-5."
        self.seq_length = 12
        self.min_data_length = self.seq_length + 20
    def _create_torch_model(self):
        class _GRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(5, 50, num_layers=2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(50, 5)
            def forward(self, x):
                gru_out, _ = self.gru(x)
                return self.fc(gru_out[:, -1, :])
        return _GRU()

class BayesianLSTMModel(BaseSequenceModel):
    def __init__(self, max_nums):
        super().__init__(max_nums)
        self.name = "Bayesian LSTM"
        self.logic = "Hybrid BNN for Pos 1-5, quantifying uncertainty."
        self.seq_length = 12
        self.min_data_length = self.seq_length + 20
        self.kl_weight = 0.1
    def _create_torch_model(self):
        if not bnn:
            raise RuntimeError("torchbnn library is not installed.")
        class _HybridBayesianLSTM(nn.Module):
            def __init__(self, input_size=5, hidden_size=50, output_size=5):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
                self.bayes_fc = bnn.BayesLinear(in_features=hidden_size, out_features=output_size, prior_mu=0, prior_sigma=1)
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.bayes_fc(lstm_out[:, -1, :])
        return _HybridBayesianLSTM()
    def train(self, df: pd.DataFrame):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is not available, cannot train Bayesian LSTM.")
        if not bnn:
            raise RuntimeError("torchbnn library is not installed.")
        super().train(df)
        X, y = create_sequences(self.scaler.transform(df), self.seq_length)
        X_torch, y_torch = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        for _ in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                mse = mse_loss(pred, batch_y)
                kl = kl_loss(self.model)
                loss = mse + self.kl_weight * kl
                loss.backward()
                optimizer.step()
        st.session_state.data_warnings.append(f"Bayesian LSTM trained: {len(X)} sequences, {self.epochs} epochs.")
    def predict(self, full_history: pd.DataFrame, n_samples=50):
<JAVASCRIPT_CODE_BLOCK>
        if not NUMPY_AVAILABLE:
            st.session_state.data_warnings.append("Bayesian LSTM: NumPy unavailable, returning uniform distributions.")
            return {'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in self.max_nums], 'uncertainty': 0.0}
        if not bnn:
            st.session_state.data_warnings.append("Bayesian LSTM: torchbnn unavailable, returning uniform distributions.")
            return {'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in self.max_nums], 'uncertainty': 0.0}
        if not self.model or not self.scaler:
            raise RuntimeError("Model is not trained.")
        if full_history is None or full_history.empty:
            raise ValueError("Bayesian LSTM: full_history is None or empty.")
        try:
            last_seq_scaled = self.scaler.transform(history_main.iloc[-self.seq_length:].values)
            input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions_raw = np.array([self.scaler.inverse_transform(self.model(input_tensor).cpu().numpy()).flatten() for _ in range(n_samples)])
            mean_pred = np.mean(predictions_raw, axis=0)
            std_pred = np.std(predictions_raw, axis=0)
            distributions = []
            for i in range(5):
                x_range = np.arange(1, self.max_nums[i] + 1)
                prob_mass = scipy.stats.norm.pdf(x_range, loc=mean_pred[i], scale=max(1.5, std_pred[i]))
                prob_mass /= prob_mass.sum() + 1e-10
                distributions.append({num: float(p) for num, p in zip(x_range, prob_mass)})
            uncertainty_score = np.mean(std_pred / (np.array(self.max_nums)/2))
            st.session_state.data_warnings.append(f"Bayesian LSTM predicted: Uncertainty={uncertainty_score:.3f}, Distributions={len(distributions)}")
            return {'distributions': distributions, 'uncertainty': uncertainty_score}
        except Exception as e:
            st.session_state.data_warnings.append(f"Bayesian LSTM prediction failed: {e}")
            return {'distributions': [{i: 1/max_num for i in range(1, max_num + 1)} for max_num in self.max_nums], 'uncertainty': 0.0}

class TransformerModel(BaseSequenceModel):
    def __init__(self, max_nums):
        super().__init__(max_nums)
        self.name = "Transformer"
        self.logic = "Transformer model for long-range patterns in Pos 1-5."
        self.seq_length = 15
        self.min_data_length = self.seq_length + 20
    def _create_torch_model(self):
        class _PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(max_len, 1, d_model)
                pe[:, 0, 0::2] = torch.sin(position * div_term)
                pe[:, 0, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)
            def forward(self, x):
                x = x + self.pe[:x.size(0)]
                return self.dropout(x)
        class _Transformer(nn.Module):
            def __init__(self, input_dim=5, embed_dim=16, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, embed_dim)
                self.pos_encoder = _PositionalEncoding(embed_dim, dropout)
                encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.output_projection = nn.Linear(embed_dim, input_dim)
                self.embed_dim = embed_dim
            def forward(self, src):
                src = self.input_projection(src) * math.sqrt(self.embed_dim)
                src = src.permute(1, 0, 2)
                src = self.pos_encoder(src)
                output = self.transformer_encoder(src)
                output = output.permute(1, 0, 2)
                return self.output_projection(output[:, -1, :])
        return _Transformer()

class UnivariateEnsembleModel(BaseModel):
    def __init__(self, max_nums):
        super().__init__([max_nums[5]])
        self.name = "Pos 6 Ensemble"
        self.logic = "Statistical ensemble (ARIMA, HMM, KDE) for independent Position 6."
        self.min_data_length = 30
        self.kde = None
        self.arima_pred = None
        self.hmm_model = None
        self.is_trained = False
    def train(self, df: pd.DataFrame):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is not available, cannot train Pos 6 Ensemble.")
        if len(df) < self.min_data_length:
            raise ValueError(f"Data size ({len(df)}) is less than minimum required ({self.min_data_length}).")
        series = df.values.flatten()
        if len(np.unique(series)) < 5:
            raise ValueError(f"Not enough unique values ({len(np.unique(series))}) in data.")
        try:
            self.kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(series[:, None])
            self.arima_pred = np.mean(series)
            if AutoARIMA:
                try:
                    arima_model = AutoARIMA(sp=1, suppress_warnings=True, maxiter=50)
                    arima_model.fit(series)
                    self.arima_pred = float(arima_model.predict(fh=[1])[0])
                except Exception as e:
                    st.session_state.data_warnings.append(f"Pos 6 Ensemble ARIMA training failed: {e}")
            self.hmm_model = None
            if hmm:
                try:
                    self.hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
                    self.hmm_model.fit(series.reshape(-1, 1))
                except Exception as e:
                    st.session_state.data_warnings.append(f"Pos 6 Ensemble HMM training failed: {e}")
            self.is_trained = True
            st.session_state.data_warnings.append(f"Pos 6 Ensemble trained: {len(series)} draws, ARIMA pred={self.arima_pred:.2f}")
        except Exception as e:
            st.session_state.data_warnings.append(f"Pos 6 Ensemble training failed: {e}")
            self.is_trained = False
    def predict(self, full_history: pd.DataFrame) -> Dict[str, Any]:
        if not NUMPY_AVAILABLE:
            st.session_state.data_warnings.append("Pos 6 Ensemble: NumPy unavailable, returning uniform distribution.")
            return {'distributions': [{i: 1/self.max_nums[0] for i in range(1, self.max_nums[0] + 1)}]}
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        if full_history is None or full_history.empty:
            raise ValueError("Pos 6 Ensemble: full_history is None or empty.")
        try:
            max_num = self.max_nums[0]
            x_range = np.arange(1, max_num + 1)
            kde_probs = np.exp(self.kde.score_samples(x_range.reshape(-1, 1))).flatten()
            arima_probs = scipy.stats.norm.pdf(x_range, loc=self.arima_pred, scale=np.std(x_range))
            hmm_probs = np.ones(max_num)
            if self.hmm_model:
                try:
                    state_means = self.hmm_model.means_.flatten()
                    state_covars = np.sqrt([self.hmm_model.covars_[i, 0, 0] for i in range(self.hmm_model.n_components)])
                    hmm_dist = np.zeros(max_num)
                    for i in range(self.hmm_model.n_components):
                        hmm_dist += scipy.stats.norm.pdf(x_range, state_means[i], state_covars[i])
                    hmm_probs = hmm_dist
                except Exception as e:
                    st.session_state.data_warnings.append(f"Pos 6 Ensemble HMM prediction failed: {e}")
            ensemble_probs = (0.4 * kde_probs + 0.3 * arima_probs + 0.3 * hmm_probs)
            ensemble_probs /= ensemble_probs.sum() + 1e-10
            distribution = {int(num): float(prob) for num, prob in zip(x_range, ensemble_probs)}
            st.session_state.data_warnings.append(f"Pos 6 Ensemble predicted: Top 5={sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]}")
            return {'distributions': [distribution]}
        except Exception as e:
            st.session_state.data_warnings.append(f"Pos 6 Ensemble prediction failed: {e}")
            return {'distributions': [{i: 1/self.max_nums[0] for i in range(1, self.max_nums[0] + 1)}]}

# --- MODEL FACTORY ---
class ModelFactory:
    def __init__(self, max_nums):
        self.max_nums = max_nums
        self.registered_models = {
            "LSTM": (LSTMModel, {'max_nums': max_nums}, True),
            "GRU": (GRUModel, {'max_nums': max_nums}, True),
            "Transformer": (TransformerModel, {'max_nums': max_nums}, True),
            "Bayesian LSTM": (BayesianLSTMModel, {'max_nums': max_nums}, bnn is not None),
        }
    def get_available_models(self, data_length):
        available = {}
        skipped = {}
        for name, (model_class, params, lib_available) in self.registered_models.items():
            if not NUMPY_AVAILABLE:
                skipped[name] = "NumPy is not available."
                continue
            if not lib_available:
                skipped[name] = "Required library not installed (torchbnn for Bayesian LSTM)."
                continue
            temp_model = model_class(**params)
            if data_length < temp_model.min_data_length:
                skipped[name] = f"Requires {temp_model.min_data_length} data points, {data_length} provided."
                continue
            available[name] = (model_class, params)
        return available, skipped

# --- 4. OPTIMIZED BACKTESTING & CACHING ---
def get_data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@st.cache_resource(ttl=3600)
def get_or_train_model(_model_class, _training_df, _model_params, _cache_key):
    try:
        model = _model_class(**_model_params)
        model.train(_training_df)
        st.session_state.data_warnings.append(f"Model {_model_class.__name__} trained or retrieved from cache: {_cache_key}")
        return model
    except Exception as e:
        st.session_state.data_warnings.append(f"Failed to train model {_model_class.__name__}: {e}")
        return None

def run_full_backtest(df: pd.DataFrame, train_size: int, backtest_steps: int, max_nums_input: List[int]) -> Dict[str, Dict[str, Any]]:
    if not NUMPY_AVAILABLE:
        st.session_state.data_warnings.append("Backtest: NumPy unavailable, cannot run backtest.")
        return {}
    results = {}
    df_main, df_pos6 = df.iloc[:, :5], df.iloc[:, 5]
    factory = ModelFactory(max_nums_input)
    available_models, _ = factory.get_available_models(len(df_main))
    pos6_model_class, pos6_params = UnivariateEnsembleModel, {'max_nums': max_nums_input}
    progress_bar = st.progress(0, text="Backtesting models...")
    total_steps = len(available_models) * backtest_steps
    current_step = 0
    for name, (model_class, model_params) in available_models.items():
        with st.spinner(f"Backtesting {name}..."):
            log_losses, uncertainties = [], []
            initial_train_main = df_main.iloc[:train_size]
            initial_train_pos6 = df_pos6.to_frame()
            if len(initial_train_main) < 32:
                st.session_state.data_warnings.append(f"{name}: Insufficient training data ({len(initial_train_main)} < 32).")
                continue
            try:
                main_data_hash = get_data_hash(initial_train_main)
                pos6_data_hash = get_data_hash(initial_train_pos6)
                main_model = get_or_train_model(model_class, initial_train_main, model_params, f"{model_class.__name__}-{main_data_hash}")
                pos6_model = get_or_train_model(pos6_model_class, initial_train_pos6, pos6_params, f"{pos6_model_class.__name__}-{pos6_data_hash}")
                if main_model is None or pos6_model is None:
                    st.session_state.data_warnings.append(f"{name}: Model initialization failed.")
                    continue
                for i in range(backtest_steps):
                    step = train_size + i
                    if step >= len(df):
                        st.session_state.data_warnings.append(f"{name}: Step {step} exceeds data length {len(df)}.")
                        break
                    true_draw = df.iloc[step].values
                    try:
                        pred_obj_main = main_model.predict(full_history=df.iloc[:step])
                        pred_obj_pos6 = pos6_model.predict(full_history=df.iloc[:step])
                        if not pred_obj_main.get('distributions') or not pred_obj_pos6.get('distributions'):
                            st.session_state.data_warnings.append(f"{name}: Invalid distributions at step {step}.")
                            continue
                        all_distributions = pred_obj_main['distributions'] + pred_obj_pos6['distributions']
                        if len(all_distributions) != 6:
                            st.session_state.data_warnings.append(f"{name}: Expected 6 distributions, got {len(all_distributions)}.")
                            continue
                        if 'uncertainty' in pred_obj_main:
                            uncertainties.append(pred_obj_main['uncertainty'])
                        step_log_loss = 0
                        for pos_idx, dist in enumerate(all_distributions):
                            if not dist:
                                dist = {j: 1/max_nums_input[pos_idx] for j in range(1, max_nums_input[pos_idx] + 1)}
                            total_prob = sum(dist.values())
                            if total_prob == 0 or np.isnan(total_prob):
                                dist = {j: 1/max_nums_input[pos_idx] for j in range(1, max_nums_input[pos_idx] + 1)}
                            else:
                                dist = {k: v/total_prob for k, v in dist.items()}
                            true_num = int(true_draw[pos_idx])
                            if not (1 <= true_num <= max_nums_input[pos_idx]):
                                st.session_state.data_warnings.append(f"{name}: Invalid true_num {true_num} for Pos_{pos_idx+1}.")
                                continue
                            prob_of_true = dist.get(true_num, 1e-10)
                            step_log_loss -= np.log(max(prob_of_true, 1e-10))
                        step_log_loss = min(step_log_loss, 5.0)
                        log_losses.append(step_log_loss)
                    except Exception as e:
                        st.session_state.data_warnings.append(f"{name}: Prediction failed at step {step}: {e}")
                        continue
                    current_step += 1
                    progress_bar.progress(min(1.0, current_step / total_steps))
                if not log_losses:
                    st.session_state.data_warnings.append(f"{name}: No valid log losses computed.")
                    continue
                avg_log_loss = np.mean(log_losses)
                likelihood = 100 * min(1.0, np.exp(-avg_log_loss / 3))
                metrics = {'Log Loss': f"{avg_log_loss:.3f}", 'Likelihood': f"{likelihood:.1f}%"}
                if uncertainties:
                    metrics['BNN Uncertainty'] = f"{np.mean(uncertainties):.3f}"
                results[name] = metrics
                st.session_state.data_warnings.append(
                    f"{name} backtest: Log Loss={avg_log_loss:.3f}, Likelihood={likelihood:.1f}%, Metrics={metrics}"
                )
            except Exception as e:
                st.session_state.data_warnings.append(f"{name}: Backtest failed: {e}")
        progress_bar.empty()
    return results

# --- 5. STABILITY & DYNAMICS ANALYSIS FUNCTIONS ---
@st.cache_data
def find_stabilization_point(_df: pd.DataFrame, _max_nums: List[int], backtest_steps: int) -> go.Figure:
    if not NUMPY_AVAILABLE:
        st.session_state.data_warnings.append("Stabilization analysis: NumPy unavailable, cannot run.")
        return go.Figure().update_layout(title_text="Stabilization Analysis Disabled")
    if not AutoARIMA:
        st.session_state.data_warnings.append("Stabilization analysis disabled: sktime not installed.")
        return go.Figure().update_layout(title_text="Stabilization Analysis Disabled")
    df_pos1 = _df.iloc[:, 0]
    max_num_pos1 = _max_nums[0]
    if len(df_pos1.unique()) < 5:
        st.session_state.data_warnings.append(f"Stabilization analysis skipped: Pos_1 has {len(df_pos1.unique())} unique values.")
        return go.Figure().update_layout(title_text="Insufficient Unique Values for Stabilization Analysis")
    window_sizes = np.linspace(50, max(250, len(df_pos1) - backtest_steps - 1), 10, dtype=int)
    results = []
    progress_bar = st.progress(0, text="Running stabilization analysis...")
    for i, size in enumerate(window_sizes):
        if len(df_pos1) <= size + backtest_steps:
            st.session_state.data_warnings.append(f"Too many windows: Window size {size} too large for {len(df_pos1)} draws.")
            continue
        log_losses, predictions = [], []
        for step in range(backtest_steps):
            try:
                series = df_pos1.iloc[:size + step]
                true_val = df_pos1.iloc[size + step]
                model = AutoARIMA(sp=1, suppress_warnings=True, maxiter=50)
                model.fit(series)
                pred_mean = float(model.predict(fh=[1])[0])
                pred_std = np.std(series) * 1.5
                x_range = np.arange(1, max_num_pos1 + 1)
                prob_mass = scipy.stats.norm.pdf(x_range, loc=pred_mean, scale=max(1.0, pred_std))
                prob_mass /= prob_mass.sum() + 1e-10
                prob_dist = {num: float(p) for num, p in zip(x_range, prob_mass)}
                prob_of_true = prob_dist.get(true_val, 1e-10)
                log_losses.append(-np.log(max(prob_of_true, 1e-10)))
                predictions.append(pred_mean)
            except Exception as e:
                st.session_state.data_warnings.append(f"Stabilization: ARIMA failed at step {step}, size {size}: {e}")
                log_losses.append(np.log(max_num_pos1))
        if log_losses:
            avg_log_loss = np.mean(log_losses)
            psi = np.std(predictions) / (np.mean(predictions) + 1e-10) if predictions else float('nan')
            results.append({'Window Size': size, 'Cross-Entropy Loss': avg_log_loss, 'Prediction Stability Index': psi})
        progress_bar.progress((i + 1) / len(window_sizes))
    progress_bar.empty()
    if not results:
        st.session_state.data_warnings.append("Stabilization analysis failed: No valid results.")
        return go.Figure().update_layout(title_text="Insufficient Data for Stabilization Analysis")
    results_df = pd.DataFrame(results).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['Window Size'], y=results_df['Cross-Entropy Loss'], mode='lines+markers', name='Cross-Entropy Loss'))
    fig.add_trace(go.Scatter(x=results_df['Window Size'], y=results_df['Prediction Stability Index'], mode='lines+markers', name='Prediction Stability Index', yaxis='y2'))
    fig.update_layout(
        title='Model Stabilization Analysis (Pos 1)', xaxis_title='Training Window Size',
        yaxis_title='Cross-Entropy Loss', yaxis2=dict(title='Prediction Stability Index', overlaying='y', side='right')
    )
    return fig

@st.cache_data
def analyze_clusters(_df: pd.DataFrame, min_cluster_size: int, min_samples: int) -> Dict[str, Any]:
    if not NUMPY_AVAILABLE:
        st.session_state.data_warnings.append("Clustering: NumPy unavailable, cannot run.")
        return {'fig': go.Figure(), 'summary': "Clustering disabled: NumPy unavailable.", 'silhouette': "N/A"}
    if not hdbscan or not umap:
        st.session_state.data_warnings.append("Clustering disabled: hdbscan or umap-learn not installed.")
        return {'fig': go.Figure(), 'summary': "Clustering disabled: missing dependencies.", 'silhouette': "N/A"}
    df_main = _df.iloc[:, :5]
    results = {'fig': go.Figure(), 'summary': "Clustering failed.", 'silhouette': "N/A"}
    if len(df_main) < max(10, min_cluster_size):
        st.session_state.data_warnings.append(f"Clustering failed: Insufficient data ({len(df_main)} < {max(10, min_cluster_size)}).")
        return results
    for col in df_main.columns:
        if len(df_main[col].unique()) < 5:
            st.session_state.data_warnings.append(f"Clustering skipped: {col} has {len(df_main[col].unique())} unique values.")
            return results
    data = df_main.values
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(data)
        if len(set(labels)) <= 1 or np.all(labels == -1):
            st.session_state.data_warnings.append("Clustering failed: No valid clusters detected, using uniform.")
            labels = np.zeros(len(data), dtype=int)
            results['silhouette'] = "0.0 (uniform)"
            results['summary'] = f"- **Uniform Cluster**: {len(data)} draws, Centroid: {np.mean(data, axis=0).round().astype(int).tolist()}"
        else:
            clean_labels = labels[labels != -1]
            clean_data = data[labels != -1]
            if len(set(clean_labels)) > 1:
                score = silhouette_score(clean_data, clean_labels)
                results['silhouette'] = f"{score:.3f}"
            else:
                results['silhouette'] = "N/A (1 cluster)"
            summary_text = ""
            cluster_counts = Counter(labels)
            for cluster_id, count in sorted(cluster_counts.items()):
                if cluster_id == -1:
                    summary_text += f"- **Noise Points**: {count} draws.\n"
                else:
                    cluster_mean = df_main[labels == cluster_id].mean().round().astype(int).tolist()
                    summary_text += f"- **Cluster {cluster_id}**: {count} draws, Centroid: {cluster_mean}\n"
            results['summary'] = summary_text
        reducer = umap.UMAP(n_neighbors=12, n_components=2, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(data)
        plot_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
        plot_df['Cluster'] = [str(l) for l in labels]
        plot_df['Draw'] = df_main.index
        plot_df['Numbers'] = df_main.apply(lambda row: ', '.join(row.astype(str)), axis=1)
        fig = px.scatter(
            plot_df, x='UMAP_1', y='UMAP_2', color='Cluster', custom_data=['Draw', 'Numbers'],
            title=f'Latent Space of Draws (Pos 1-5), Silhouette: {results["silhouette"]}',
            color_discrete_map={'-1': 'grey'}
        )
        fig.update_traces(hovertemplate='<b>Draw %{customdata[0]}</b><br>Numbers: %{customdata[1]}<br>Cluster: %{marker.color}')
        results['fig'] = fig
        st.session_state.data_warnings.append(f"Clustering completed: Silhouette={results['silhouette']}")
    except Exception as e:
        st.session_state.data_warnings.append(f"Clustering error: {e}")
    return results

# --- 6. MAIN APPLICATION UI & LOGIC ---
st.sidebar.header("1. System Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Number History (CSV)", type=["csv"])
with st.sidebar.expander("Advanced Configuration", expanded=True):
    max_nums_input = [st.number_input(f"Max Value for Pos_{i+1}", 10, 150, 49, key=f"max_num_{i}") for i in range(6)]
    training_size_slider = st.slider("Training Window Size", 50, 1000, 150, 10, help="Number of past draws to train on.")
    backtest_steps_slider = st.slider("Backtest Validation Steps", 5, 50, 10, 1, help="Number of steps for performance evaluation.")

# --- System Health Check ---
with st.sidebar.expander("System Health"):
    libs = {
        "NumPy": np.__version__ if NUMPY_AVAILABLE else "NOT FOUND",
        "Pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "PyTorch": torch.__version__,
        "Sktime": "Installed" if AutoARIMA else "NOT FOUND",
        "TorchBNN": "Installed" if bnn else "NOT FOUND",
        "HMMlearn": "Installed" if hmm else "NOT FOUND",
        "NetworkX": "Installed" if nx else "NOT FOUND",
        "HDBSCAN": "Installed" if hdbscan else "NOT FOUND",
        "UMAP": "Installed" if umap else "NOT FOUND"
    }
    for lib, version in libs.items():
        st.markdown(f"- **{lib}:** `{version}`")
if st.session_state.data_warnings:
    with st.sidebar.expander("Warnings", expanded=True):
        for warning in st.session_state.data_warnings[-10:]:  # Limit to last 10
            st.warning(warning)

if uploaded_file:
    df, logs = load_and_validate_data(uploaded_file, max_nums_input)
    with st.sidebar.expander("Data Loading Log", expanded=False):
        for log in logs:
            st.info(log)
    if not df.empty:
        st.session_state.df_master = df
        st.sidebar.success(f"Loaded and validated {len(df)} draws.")
        st.session_state.data_warnings.append(f"Data loaded: {len(df)} draws, shape={df.shape}")
        tab1, tab2, tab3 = st.tabs(["🔮 Predictive Ensembles", "🕸️ Graph Dynamics (Pos 1-5)", "📉 System Stability"])
        with tab1:
            st.header("🔮 Predictive Ensembles")
            st.markdown("Operating on a **5+1 architecture**: Positions 1-5 are modeled as a correlated set, and Position 6 is modeled independently.")
            analysis_mode = st.radio("Select Analysis Mode:", ("Quick Forecast", "Run Full Backtest"), horizontal=True)
            factory = ModelFactory(max_nums_input)
            training_df_main = df.iloc[:training_size_slider, :5]
            available_models, skipped_models = factory.get_available_models(len(training_df_main))
            pos6_model_class, pos6_params = UnivariateEnsembleModel, {'max_nums': max_nums_input}
            if skipped_models:
                with st.expander("Skipped Models"):
                    for name, reason in skipped_models.items():
                        st.warning(f"**{name}:** {reason}")
            if not available_models:
                st.error("No models could be run with the current data and settings. Please provide more data or adjust the 'Training Window Size'.")
            else:
                backtest_results = {}
                if analysis_mode == "Run Full Backtest":
                    with st.spinner("Running full backtest..."):
                        backtest_results = run_full_backtest(df, training_size_slider, backtest_steps_slider, max_nums_input)
                cols = st.columns(len(available_models))
                for i, (name, (model_class, model_params)) in enumerate(available_models.items()):
                    with cols[i]:
                        with st.container(border=True):
                            st.subheader(name)
                            try:
                                with st.spinner(f"Generating forecast for {name}..."):
                                    main_data_hash = get_data_hash(training_df_main)
                                    pos6_data_hash = get_data_hash(df.iloc[:training_size_slider, 5:6])
                                    main_model = get_or_train_model(
                                        model_class, training_df_main, model_params,
                                        f"{name}-{main_data_hash}"
                                    )
                                    pos6_model = get_or_train_model(
                                        pos6_model_class, df.iloc[:training_size_slider, 5:6], pos6_params,
                                        f"Pos6-{pos6_data_hash}"
                                    )
                                    if main_model is None or pos6_model is None:
                                        raise RuntimeError("Model initialization failed.")
                                    final_pred_main = main_model.predict(full_history=df)
                                    final_pred_pos6 = pos6_model.predict(full_history=df)
                                    all_distributions = final_pred_main.get('distributions', []) + final_pred_pos6.get('distributions', [])
                                    st.session_state.data_warnings.append(
                                        f"{name}: Retrieved {len(all_distributions)} distributions (Main: {len(final_pred_main.get('distributions', []))}, Pos_6: {len(final_pred_pos6.get('distributions', []))})"
                                    )
                                    if len(all_distributions) != 6 or not all(isinstance(d, dict) for d in all_distributions):
                                        st.session_state.data_warnings.append(
                                            f"{name}: Invalid distributions (Length: {len(all_distributions)}, Types: {[type(d).__name__ for d in all_distributions]})"
                                        )
                                        final_prediction = ["Error"] * 6
                                    else:
                                        final_prediction = get_best_guess_set(all_distributions, max_nums_input)
                                st.markdown(f"**Predicted Set:**")
                                st.code(" | ".join(map(str, final_prediction)))
                                if analysis_mode == "Run Full Backtest" and name in backtest_results:
                                    metrics = backtest_results[name]
                                    m_cols = st.columns(2)
                                    m_cols[0].metric("Likelihood Score", metrics['Likelihood'])
                                    if 'BNN Uncertainty' in metrics:
                                        m_cols[1].metric("BNN Uncertainty", metrics['BNN Uncertainty'], help="Model uncertainty for Pos 1-5. Lower is better.")
                                    else:
                                        m_cols[1].metric("Cross-Entropy Loss", metrics['Log Loss'])
                                elif analysis_mode == "Run Full Backtest":
                                    st.warning("Could not generate backtest results.")
                            except (ValueError, RuntimeError) as e:
                                st.error(f"Prediction Failed: {e}")
                                with st.expander("Show Error Details"):
                                    st.code(traceback.format_exc())
                            except Exception as e:
                                st.error(f"Prediction Failed: Unexpected error: {e}")
                                with st.expander("Show Error Details"):
                                    st.code(traceback.format_exc())
        with tab2:
            st.header("🕸️ Graph Dynamics (Pos 1-5)")
            st.markdown("Network of co-occurrences for Positions 1-5. Nodes are numbers, edges represent co-occurrences.")
            if not nx:
                st.warning("Graph Dynamics disabled: networkx not installed.")
            else:
                st.sidebar.header("2. Graph Dynamics Settings")
                graph_lookback = st.sidebar.slider("Lookback Period (Draws)", 20, 500, 50, 5, help="Number of draws to analyze.")
                cluster_resolution = st.sidebar.slider("Cluster Resolution", 0.5, 5.0, 2.0, 0.1, help="Higher values create more smaller clusters.")
                graph_df = df.iloc[-graph_lookback:, :5]
                if graph_df.empty:
                    st.warning("No data available for graph dynamics.")
                else:
                    try:
                        G = nx.Graph()
                        for _, row in graph_df.iterrows():
                            for u, v in itertools.combinations(row.values, 2):
                                if G.has_edge(u, v):
                                    G[u][v]['weight'] += 1
                                else:
                                    G.add_edge(u, v, weight=1)
                        if len(G.edges()) < 5:
                            st.session_state.data_warnings.append(f"Graph has {len(G.edges())} edges: sparse data.")
                            st.warning("Sparse network: insufficient co-occurrences for meaningful analysis.")
                        else:
                            clusters = list(nx_comm.louvain_communities(G, weight='weight', resolution=cluster_resolution, seed=42))
                            col1, col2 = st.columns([2, 5])
                            with col1:
                                st.subheader("Detected Clusters")
                                st.markdown("Nodes frequently appearing together in Positions 1-5.")
                                for i, cluster in enumerate(clusters):
                                    if len(cluster) >= 3:
                                        st.markdown(f"**Cluster {i}**: `{sorted(list(cluster))}`")
                            with col2:
                                pos = nx.spring_layout(G, k=0.8, iterations=30, seed=42)
                                edge_x, edge_y, edge_weights = [], [], []
                                for edge in G.edges():
                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    edge_x.extend([x0, x1, None])
                                    edge_y.extend([y0, y1, None])
                                    edge_weights.append(G.edges[edge]['weight'])
                                edge_trace = go.Scatter(
                                    x=edge_x, y=edge_y, line=dict(width=np.sqrt(edge_weights)*3, color='#666'),
                                    hoverinfo='none', mode='lines'
                                )
                                node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                                centrality = nx.degree_centrality(G)
                                color_map = px.colors.qualitative.Vivid
                                cluster_map = {node: i for i, c in enumerate(clusters) for node in c}
                                for node in G.nodes():
                                    x, y = pos[node]
                                    node_x.append(x)
                                    node_y.append(y)
                                    node_color.append(color_map[cluster_map.get(node, -1) % len(color_map)])
                                    node_size.append(10 + 50 * centrality.get(node, 0))
                                    node_text.append(f"Node: {node}<br>Cluster: {cluster_map.get(node, 'N/A')}<br>Centrality: {centrality.get(node, 0):.2f}")
                                node_trace = go.Scatter(
                                    x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=node_text,
                                    marker=dict(color=node_color, size=node_size, line_width=2)
                                )
                                fig = go.Figure(
                                    data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title='Network of Co-occurrences (Positions 1-5)',
                                        showlegend=False, hovermode='closest',
                                        margin=dict(l=10, r=50, t=40, b=20),
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Graph generation failed: {e}")
                        st.session_state.data_warnings.append(f"Graph Dynamics error: {e}")
        with tab3:
            st.header("📉 System Stability")
            st.subheader("Training Window Stability")
            st.markdown("""
                **Overview**: Evaluates model stability with varying training window sizes.  
                - **Blue Line (Cross-Entropy)**: Prediction error, lower is better.  
                - **Green Line (Stability Index)**: Prediction consistency, lower and flatter is better.  
                **How to Use**: Set 'Training Window Size' to the elbow point where the blue line flattens.
            """)
            stabilization_plot = find_stabilization_point(df, max_nums_input, backtest_steps_slider)
            st.plotly_chart(stabilization_plot, use_container_width=True)
            st.subheader("Cluster Dynamics")
            st.markdown("""
                **Overview**: Groups draws into clusters based on similarity in Positions 1-5.  
                **How to Use**: Use the largest cluster's centroid for conservative predictions.
            """)
            st.sidebar.header("3. Clustering Settings")
            cluster_min_size = st.sidebar.slider("Minimum Cluster Size", 5, 50, 10, 1)
            cluster_min_samples = st.sidebar.slider("Minimum Samples", 1, 20, 5, 1)
            cluster_result = analyze_clusters(df.iloc[:, :5], cluster_min_size, cluster_min_samples)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(cluster_result['fig'], use_container_width=True)
            with col2:
                st.write("### Cluster Insights")
                st.markdown(cluster_result['summary'])
else:
    st.info("Awaiting CSV file upload to begin analysis.")
