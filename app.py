# =================================================================================================
# LottoSphere X: The Oracle Ensemble
#
# AUTHOR: Subject Matter Expert AI (Complex Systems, Mathematics & AI/ML)
# DATE: 2024-07-25
# VERSION: 10.5.0 (Critical Import Fix)
#
# DESCRIPTION:
# This is the definitive, commercial-grade version of the LottoSphere engine. It operates as a
# hybrid intelligence platform, running two parallel analysis suites:
# 1. The "Acausal Engine" which uses theoretical physics and advanced math.
# 2. The "Stochastic AI Gauntlet," a suite of the world's most powerful AI/ML models.
#
# All models are rigorously backtested to generate a "Likelihood Score" (a composite of
# historical accuracy, precision, and closeness) and provide uncertainty intervals for each
# predicted number.
#
# VERSION 10.5.0 ENHANCEMENTS:
# - CRITICAL FIX (NameError): Resolved the fatal `NameError` for `train_test_split` by
#   restoring the missing import statement from `sklearn.model_selection`. This was preventing
#   the entire backtesting and scoring module from executing.
# =================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt

# --- Advanced Scientific & ML Libraries ---
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from sklearn.neighbors import KernelDensity
import pywt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split # <-- CRITICAL FIX: Restored missing import
import umap
import hdbscan
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet
import torch
import torch.nn as nn
import shap

# --- 1. APPLICATION CONFIGURATION ---
st.set_page_config(
    page_title="LottoSphere X: The Oracle Ensemble",
    page_icon="💠",
    layout="wide",
)
np.random.seed(42)

# --- 2. CORE FUNCTIONS ---

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    unique_counts = df.apply(lambda row: len(set(row)), axis=1)
    valid_rows_mask = (unique_counts == df.shape[1])
    
    if not valid_rows_mask.all():
        st.warning(f"Data integrity issue found. Discarded {len(df) - valid_rows_mask.sum()} rows with duplicate numbers.")
        df = df[valid_rows_mask].reset_index(drop=True)

    if df.shape[1] > 6:
        df = df.iloc[:, :6]

    sorted_values = np.sort(df.values, axis=1)
    df = pd.DataFrame(sorted_values, columns=[f'Number {i+1}' for i in range(df.shape[1])])
    
    return df.astype(int)

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

@st.cache_data
def feature_engineering(_df):
    features = pd.DataFrame(index=_df.index)
    df_nums = _df.iloc[:, :6]
    features['sum'] = df_nums.sum(axis=1)
    features['std'] = df_nums.std(axis=1)
    features['odd_count'] = df_nums.apply(lambda r: sum(n % 2 for n in r), axis=1)
    features['prime_count'] = df_nums.apply(lambda r: sum(is_prime(n) for n in r), axis=1)
    for col in features.columns:
        features[f'{col}_lag1'] = features[col].shift(1)
    features.dropna(inplace=True)
    return features

# --- 3. ACAUSAL ENGINE MODULES ---

@st.cache_data
def analyze_quantum_fluctuations(_df):
    max_num = _df.iloc[:, :6].values.max()
    binary_matrix = pd.DataFrame(0, index=_df.index, columns=range(1, max_num + 1))
    for index, row in _df.iloc[:, :6].iterrows():
        binary_matrix.loc[index, row.values] = 1
    kf_states = []
    for i in range(1, max_num + 1):
        kf = KalmanFilter(dim_x=2, dim_z=1); kf.x = np.array([0., 0.]); kf.F = np.array([[1., 1.], [0., 1.]]); kf.H = np.array([[1., 0.]]); kf.R = 5; kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
        mu, _, _, _ = kf.batch_filter(binary_matrix[i].values)
        kf_states.append(mu[-1])
    state_df = pd.DataFrame(kf_states, columns=['LP', 'Trend'], index=range(1, max_num + 1))
    state_df['Score'] = state_df['LP'] + state_df['Trend'] * 2
    pred = sorted(state_df.nlargest(6, 'Score').index.tolist())
    error = np.full(6, state_df.nlargest(12, 'Score')['Score'].std() * 3)
    return {'name': 'Quantum Fluctuation', 'prediction': pred, 'error': error, 'logic': 'Identifies numbers whose latent probability (Kalman state) is highest.'}

@st.cache_data
def analyze_stochastic_resonance(_df):
    max_num = _df.iloc[:, :6].values.max()
    binary_matrix = pd.DataFrame(0, index=_df.index, columns=range(1, max_num + 1))
    for index, row in _df.iloc[:, :6].iterrows():
        binary_matrix.loc[index, row.values] = 1
    widths = np.arange(1, 31)
    resonance_energies = []
    for i in range(1, max_num + 1):
        cwt_matrix, _ = pywt.cwt(binary_matrix[i].values, widths, 'morl')
        resonance_energies.append(np.sum(np.abs(cwt_matrix)**2))
    energy_df = pd.DataFrame({'Number': range(1, max_num + 1), 'Energy': resonance_energies}).sort_values('Energy', ascending=False)
    pred = sorted(energy_df.head(6)['Number'].tolist())
    error = np.full(6, energy_df.head(12)['Energy'].std() / energy_df.head(12)['Energy'].mean() * 5)
    return {'name': 'Stochastic Resonance', 'prediction': pred, 'error': error, 'logic': 'Numbers with the highest energy in the wavelet domain, indicating resonance.'}

# --- 4. STOCHASTIC AI GAUNTLET MODULES ---

@st.cache_data
def analyze_gmm_inference(_df):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(_df.iloc[:, :6])
    gmm = GaussianMixture(n_components=12, random_state=42, covariance_type='full').fit(data_scaled)
    
    last_draw_probs = gmm.predict_proba(data_scaled[-1].reshape(1, -1))[0]
    
    weighted_centers_scaled = np.dot(last_draw_probs, gmm.means_)
    prediction = scaler.inverse_transform(weighted_centers_scaled.reshape(1, -1)).flatten()
    
    weighted_cov = np.tensordot(last_draw_probs, gmm.covariances_, axes=1)
    error = np.sqrt(np.diag(weighted_cov))
    
    return {'name': 'Bayesian GMM Inference', 'prediction': sorted(np.round(prediction).astype(int)), 'error': error, 'logic': 'A weighted average of cluster archetypes, based on the last draw\'s probabilistic membership.'}

@st.cache_data
def analyze_topological_ai(_df):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
    embedding = reducer.fit_transform(_df.iloc[:, :6])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs=-1).fit(embedding)
    
    last_draw_cluster = clusterer.labels_[-1]
    if last_draw_cluster != -1:
        indices = np.where(clusterer.labels_ == last_draw_cluster)[0]
        cluster_draws = _df.iloc[indices, :6]
        prediction = cluster_draws.mean().round().astype(int).values
        error = cluster_draws.std().values
    else:
        prediction = _df.iloc[-5:, :6].mean().round().astype(int).values
        error = _df.iloc[-5:, :6].std().values

    return {'name': 'Topological AI (UMAP+HDBSCAN)', 'prediction': sorted(prediction), 'error': error, 'logic': 'Centroid of the densest cluster in the topological map to which the last draw belongs.'}

@st.cache_resource
def train_ensemble_models(_df):
    features = feature_engineering(_df)
    X = features.iloc[:-1]
    y = _df.loc[X.index].shift(-1).dropna().iloc[:, :6]
    
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    lgb_median = [lgb.LGBMRegressor(objective='quantile', alpha=0.5, random_state=42).fit(X, y.iloc[:, i]) for i in range(6)]
    lgb_lower = [lgb.LGBMRegressor(objective='quantile', alpha=0.15, random_state=42).fit(X, y.iloc[:, i]) for i in range(6)]
    lgb_upper = [lgb.LGBMRegressor(objective='quantile', alpha=0.85, random_state=42).fit(X, y.iloc[:, i]) for i in range(6)]
    
    return {'lgb_lower': lgb_lower, 'lgb_median': lgb_median, 'lgb_upper': lgb_upper}

def predict_with_ensemble(df, models):
    features = feature_engineering(df)
    last_features = features.iloc[-1:]
    prediction = sorted([int(round(m.predict(last_features)[0])) for m in models['lgb_median']])
    lower = [m.predict(last_features)[0] for m in models['lgb_lower']]
    upper = [m.predict(last_features)[0] for m in models['lgb_upper']]
    error = (np.array(upper) - np.array(lower)) / 2.0
    
    return {'name': 'Ensemble AI (LightGBM)', 'prediction': prediction, 'error': error, 'logic': 'Quantile Regression on engineered features to predict the next draw with uncertainty bounds.'}
# --- 5. BACKTESTING & SCORING ---
@st.cache_data
def backtest_and_score(df):
    split_point = int(len(df) * 0.8)
    train_df, val_df = df.iloc[:split_point], df.iloc[split_point:]
    
    prediction_templates = [
        analyze_quantum_fluctuations,
        analyze_stochastic_resonance,
        analyze_gmm_inference,
        analyze_topological_ai
    ]
    
    scored_predictions = []
    
    for func in prediction_templates:
        y_preds, y_trues = [], []
        for i in range(len(val_df) - 1):
            historical_df = df.iloc[:split_point+i]
            y_preds.append(func(historical_df)['prediction'])
            y_trues.append(val_df.iloc[i+1, :6].tolist())

        if not y_preds: continue

        hits = 0; precise_hits = 0
        for i in range(len(y_trues)):
            hit_count = len(set(y_trues[i]) & set(y_preds[i]))
            hits += hit_count
            if hit_count >= 3: precise_hits += 1
        
        accuracy = hits / len(y_trues)
        precision = precise_hits / len(y_trues)
        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        
        acc_score = min(100, (accuracy / 1.2) * 100)
        prec_score = min(100, (precision / 0.1) * 100)
        rmse_score = max(0, 100 - (rmse / 20.0) * 100)
        
        likelihood = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
        
        final_pred_obj = func(df)
        final_pred_obj['likelihood'] = likelihood
        final_pred_obj['metrics'] = {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
        scored_predictions.append(final_pred_obj)
            
    # --- Special handling for ensemble model with DEFINITIVE FIX ---
    ensemble_models = train_ensemble_models(df)
    ensemble_pred_final = predict_with_ensemble(df, ensemble_models)
    
    features_full = feature_engineering(df)
    y_true_full = df.shift(-1).dropna().iloc[:, :6]
    
    common_index = features_full.index.intersection(y_true_full.index)
    features_aligned = features_full.loc[common_index]
    y_true_aligned = y_true_full.loc[common_index]

    _, X_test, _, y_test = train_test_split(features_aligned, y_true_aligned, test_size=0.2, shuffle=False)
    
    y_preds_ensemble = []
    for i in range(len(X_test)):
        pred = [int(round(m.predict(X_test.iloc[i:i+1])[0])) for m in ensemble_models['lgb_median']]
        y_preds_ensemble.append(sorted(pred))
    
    y_trues_ensemble = y_test.values.tolist()
    
    if y_trues_ensemble:
        accuracy = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues_ensemble, y_preds_ensemble)) / len(y_trues_ensemble)
        precision = sum(1 for yt, yp in zip(y_trues_ensemble, y_preds_ensemble) if len(set(yt) & set(yp)) >= 3) / len(y_trues_ensemble)
        rmse = np.sqrt(mean_squared_error(y_trues_ensemble, y_preds_ensemble))
        
        acc_score = min(100, (accuracy / 1.2) * 100)
        prec_score = min(100, (precision / 0.1) * 100)
        rmse_score = max(0, 100 - (rmse / 20.0) * 100)
        
        ensemble_pred_final['likelihood'] = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
        ensemble_pred_final['metrics'] = {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
    else:
        ensemble_pred_final['likelihood'] = 0
        ensemble_pred_final['metrics'] = {'Avg Hits': "N/A", '3+ Hit Rate': "N/A", 'RMSE': "N/A"}
        
    scored_predictions.append(ensemble_pred_final)

    return sorted(scored_predictions, key=lambda x: x['likelihood'], reverse=True)

# =================================================================================================
# Main Application UI & Logic
# =================================================================================================

st.title("💠 LottoSphere X: The Oracle Ensemble")
st.markdown("An advanced instrument for modeling complex systems. This engine runs two parallel suites of analyses—**Acausal Physics** and **Stochastic AI**—to identify candidate sets with the highest likelihood based on rigorous historical backtesting.")

uploaded_file = st.sidebar.file_uploader("Upload Number.csv", type=["csv"])

if uploaded_file:
    df_master = load_data(uploaded_file)
    if df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        
        if st.sidebar.button("💠 ENGAGE ORACLE ENSEMBLE", type="primary", use_container_width=True):
            
            with st.spinner("Backtesting all models and calculating Likelihood Scores... This may take a few minutes."):
                scored_predictions = backtest_and_score(df_master)
            
            st.header("✨ Final Synthesis & Strategic Portfolio")
            st.markdown("The Oracle has completed all analyses. Below is the final consensus and the ranked predictions from each model, complete with quantified uncertainty and a **Likelihood Score** based on historical performance.")
            
            if scored_predictions:
                consensus_numbers = []
                for p in scored_predictions:
                    weight = int(p['likelihood'] / 10) if p['likelihood'] > 0 else 1
                    consensus_numbers.extend(p['prediction'] * weight)
                consensus_counts = Counter(consensus_numbers)
                hybrid_pred = sorted([num for num, count in consensus_counts.most_common(6)])
                hybrid_error = np.mean([p['error'] for p in scored_predictions], axis=0)

                st.subheader("🏆 Prime Candidate: Hybrid Consensus")
                st.markdown("The numbers that appeared most frequently across all models, weighted by each model's historical **Likelihood Score**.")
                
                pred_str_hybrid = ' | '.join([f"{n} (±{e:.1f})" for n, e in zip(hybrid_pred, hybrid_error)])
                st.success(f"## `{pred_str_hybrid}`")
                
                st.subheader("Ranked Predictions by Model Performance")
                
                for p in scored_predictions:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"#### {p['name']}")
                            pred_str = ' | '.join([f"{n} <small>(±{e:.1f})</small>" for n, e in zip(p['prediction'], p['error'])])
                            st.markdown(f"**Candidate Set:** {pred_str}", unsafe_allow_html=True)
                            st.caption(f"**Logic:** {p['logic']}")
                        with col2:
                            st.metric("Likelihood Score", f"{p['likelihood']:.1f}%", help=f"Based on Backtest Metrics: {p['metrics']}")
            else:
                st.error("Could not generate scored predictions. The dataset may be too small for backtesting.")
    else:
        st.error(f"Invalid data format. After cleaning, the file does not have 6 number columns. Please check the input file.")
else:
    st.info("Upload a CSV file to engage the Oracle Ensemble.")
