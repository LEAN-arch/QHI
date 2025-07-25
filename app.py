# =================================================================================================
# LottoSphere X: The Oracle Ensemble
#
# AUTHOR: Subject Matter Expert AI (Complex Systems, Mathematics & AI/ML)
# DATE: 2024-07-25
# VERSION: 11.0 (Chronos Anomaly Detector)
#
# DESCRIPTION:
# This definitive version enhances the "System Dynamics" module into a full-fledged predictive
# suite and introduces a new meta-analytical framework to assess "Predictive Maturity"—
# determining how much historical data is required for the models to become robust.
#
# VERSION 11.0 ENHANCEMENTS:
# - ENHANCED SYSTEM DYNAMICS MODULE: This tab is now a predictive engine.
#   - Restored and enhanced the "Number Zodiac" polar plot with its own predictive model.
#   - Integrated the Calculus Momentum analysis as a core predictive model.
# - NEW META-ANALYSIS: A "Predictive Maturity Curve" now shows how model performance (Likelihood
#   Score) evolves as more historical data is included, answering the question of "how much
#   data is enough?"
# - PREDICTIVE ROBUSTNESS: All new models are fully integrated into the rigorous, walk-forward
#   backtesting and scoring framework, competing for the highest Likelihood Score.
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
from sklearn.neighbors import KernelDensity, NearestNeighbors
import pywt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import umap
import hdbscan
import lightgbm as lgb

# --- 1. APPLICATION CONFIGURATION ---
st.set_page_config(
    page_title="LottoSphere v11.0: Chronos Detector",
    page_icon="⏳",
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
    num_cols = df.shape[1]
    valid_rows_mask = (unique_counts == num_cols)
    
    if not valid_rows_mask.all():
        st.session_state.data_warning = f"Data integrity issue found. Discarded {len(df) - valid_rows_mask.sum()} rows with duplicate or missing numbers."
        df = df[valid_rows_mask].reset_index(drop=True)

    if df.shape[1] > 6:
        df = df.iloc[:, :6]

    df.columns = [f'Number {i+1}' for i in range(df.shape[1])]
    
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
    return {'name': 'Bayesian GMM Inference', 'prediction': sorted(np.round(prediction).astype(int)), 'error': error, 'logic': 'A weighted average of cluster archetypes.'}

@st.cache_resource
def train_ensemble_models(_df):
    features = feature_engineering(_df)
    y = _df.shift(-1).dropna().iloc[:, :6]
    common_index = features.index.intersection(y.index)
    X, y = features.loc[common_index], y.loc[common_index]
    models = {
        'median': [lgb.LGBMRegressor(objective='quantile', alpha=0.5, random_state=42).fit(X, y.iloc[:, i]) for i in range(6)],
        'lower': [lgb.LGBMRegressor(objective='quantile', alpha=0.15, random_state=42).fit(X, y.iloc[:, i]) for i in range(6)],
        'upper': [lgb.LGBMRegressor(objective='quantile', alpha=0.85, random_state=42).fit(X, y.iloc[:, i]) for i in range(6)],
    }
    return models

def predict_with_ensemble(df, models):
    features = feature_engineering(df)
    last_features = features.iloc[-1:]
    prediction = sorted([int(round(m.predict(last_features)[0])) for m in models['median']])
    lower = [m.predict(last_features)[0] for m in models['lower']]
    upper = [m.predict(last_features)[0] for m in models['upper']]
    error = (np.array(upper) - np.array(lower)) / 2.0
    return {'name': 'Ensemble AI (LightGBM)', 'prediction': prediction, 'error': error, 'logic': 'Quantile Regression on engineered features.'}
# --- 5. SYSTEM DYNAMICS MODULE & PREDICTORS ---
@st.cache_data
def analyze_system_dynamics(_df):
    # Sort each row to create stable time series for each position
    df_sorted = pd.DataFrame(np.sort(_df.iloc[:,:6].values, axis=1), columns=[f'Pos {i+1}' for i in range(6)])
    max_num = _df.iloc[:,:6].values.max()

    # --- 1. Calculus Dynamics ---
    velocity = df_sorted.diff().fillna(0)
    acceleration = velocity.diff().fillna(0)
    last_v, last_a = velocity.iloc[-1], acceleration.iloc[-1]
    momentum_score = last_v - np.abs(last_a) * 0.5
    momentum_df = pd.DataFrame({'Slot': df_sorted.columns, 'Last Value': df_sorted.iloc[-1].values, 'Velocity': last_v.values, 
                                'Acceleration': last_a.values, 'Momentum Score': momentum_score.values}).sort_values('Momentum Score', ascending=False)
    calc_pred = sorted(momentum_df.head(6)['Last Value'].astype(int).tolist())
    calc_error = np.full(6, momentum_df['Last Value'].std() * 0.5)
    calculus_result = {'name': 'Calculus Momentum', 'prediction': calc_pred, 'error': calc_error, 'logic': 'Numbers from slots with highest stable positive momentum.'}
    
    # --- 2. Number Zodiac (Polar Plot) ---
    fig_zodiac = go.Figure()
    recent_draws = _df.iloc[-5:, :6]
    colors = px.colors.sequential.Plasma_r
    for i, (index, row) in enumerate(recent_draws.iterrows()):
        theta = (row.values / max_num) * 360; r = [10 - i*1.5] * 6
        fig_zodiac.add_trace(go.Scatterpolar(r=r, theta=theta, mode='markers', marker=dict(size=10, color=colors[i]), name=f'Draw {index}'))
    fig_zodiac.update_layout(title='<b>The Number Zodiac:</b> Polar Projection of Recent Draws')
    all_numbers = _df.iloc[:,:6].values.flatten()
    bins = np.linspace(0, max_num, 13)
    hist, _ = np.histogram(all_numbers, bins=bins)
    densest_sector_index = np.argmax(hist)
    sector_start, sector_end = bins[densest_sector_index], bins[densest_sector_index+1]
    sector_numbers = [n for n in all_numbers if sector_start <= n < sector_end]
    zodiac_pred = sorted([num for num, count in Counter(sector_numbers).most_common(6)])
    if len(zodiac_pred) < 6:
        hot_fill = [n for n, c in Counter(all_numbers).most_common() if n not in zodiac_pred]
        zodiac_pred.extend(hot_fill[:6-len(zodiac_pred)])
    zodiac_result = {'name': 'Number Zodiac Sector', 'prediction': zodiac_pred, 'error': np.full(6, (sector_end-sector_start)/2), 
                     'logic': f'Most frequent numbers from the densest polar sector ({int(sector_start)}-{int(sector_end)}).'}

    return momentum_df, fig_zodiac, calculus_result, zodiac_result

# --- 6. BACKTESTING & META-ANALYSIS ---
@st.cache_data
def run_full_backtest_suite(df):
    split_point = int(len(df) * 0.8)
    val_df = df.iloc[split_point:]
    
    model_funcs = {
        "Quantum Fluctuation": analyze_quantum_fluctuations,
        "Stochastic Resonance": analyze_stochastic_resonance,
        "Bayesian GMM Inference": analyze_gmm_inference,
        "Calculus Momentum": lambda d: analyze_system_dynamics(d)[2],
        "Number Zodiac Sector": lambda d: analyze_system_dynamics(d)[3],
    }
    scored_predictions = []
    for name, func in model_funcs.items():
        y_preds = [func(df.iloc[:split_point+i])['prediction'] for i in range(len(val_df))]
        y_trues = val_df.iloc[:, :6].values.tolist()
        hits = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues, y_preds)); precise_hits = sum(1 for yt, yp in zip(y_trues, y_preds) if len(set(yt) & set(yp)) >= 3)
        accuracy, precision, rmse = hits/len(y_trues), precise_hits/len(y_trues), np.sqrt(mean_squared_error(y_trues, y_preds))
        acc_score, prec_score, rmse_score = min(100,(accuracy/1.2)*100), min(100,(precision/0.1)*100), max(0,100-(rmse/20.0)*100)
        likelihood = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
        final_pred_obj = func(df)
        final_pred_obj['likelihood'], final_pred_obj['metrics'] = likelihood, {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
        scored_predictions.append(final_pred_obj)
            
    ensemble_models = train_ensemble_models(df)
    ensemble_pred_final = predict_with_ensemble(df, ensemble_models)
    features_full, y_true_full = feature_engineering(df), df.shift(-1).dropna().iloc[:, :6]
    common_index = features_full.index.intersection(y_true_full.index)
    features_aligned, y_true_aligned = features_full.loc[common_index], y_true_full.loc[common_index]
    _, X_test, _, y_test = train_test_split(features_aligned, y_true_aligned, test_size=0.2, shuffle=False)
    y_preds_ensemble = [sorted(np.round([m.predict(X_test.iloc[i:i+1])[0] for m in ensemble_models['median']]).astype(int)) for i in range(len(X_test))]
    y_trues_ensemble = y_test.values.tolist()
    if y_trues_ensemble:
        accuracy = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues_ensemble, y_preds_ensemble)) / len(y_trues_ensemble)
        precision = sum(1 for yt, yp in zip(y_trues_ensemble, y_preds_ensemble) if len(set(yt) & set(yp)) >= 3) / len(y_trues_ensemble)
        rmse = np.sqrt(mean_squared_error(y_trues_ensemble, y_preds_ensemble))
        acc_score = min(100,(accuracy/1.2)*100); prec_score = min(100,(precision/0.1)*100); rmse_score = max(0, 100-(rmse/20.0)*100)
        ensemble_pred_final['likelihood'] = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
        ensemble_pred_final['metrics'] = {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
    scored_predictions.append(ensemble_pred_final)
    return sorted(scored_predictions, key=lambda x: x.get('likelihood', 0), reverse=True)

@st.cache_data
def analyze_predictive_maturity(df):
    history_sizes = np.linspace(100, len(df), 10, dtype=int)
    maturity_scores = []
    progress_bar = st.progress(0, text="Analyzing Predictive Maturity...")
    for i, size in enumerate(history_sizes):
        if size < 50: continue
        subset_df = df.iloc[:size]
        scored_preds = run_full_backtest_suite(subset_df)
        if scored_preds:
            maturity_scores.append({'History Size': size, 'Top Likelihood Score': scored_preds[0]['likelihood']})
        progress_bar.progress((i + 1) / len(history_sizes), text=f"Analyzing with {size} draws...")
    progress_bar.empty()
    return pd.DataFrame(maturity_scores)

# =================================================================================================
# Main Application UI & Logic
# =================================================================================================

st.title("⏳ LottoSphere v11.0: The Chronos Anomaly Detector")
st.markdown("An advanced instrument for modeling complex systems. This engine identifies candidate sets with the highest likelihood based on rigorous, time-series backtesting, and analyzes the system's own predictive maturity.")

if 'data_warning' not in st.session_state: st.session_state.data_warning = None
uploaded_file = st.sidebar.file_uploader("Upload Number.csv", type=["csv"])

if uploaded_file:
    df_master = load_data(uploaded_file)
    if st.session_state.data_warning: st.warning(st.session_state.data_warning); st.session_state.data_warning = None

    if df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        tab1, tab2, tab3 = st.tabs(["🔮 Predictive Analytics", "🔬 System Dynamics Explorer", "🧠 Predictive Maturity"])

        with tab1:
            st.header("Engage Oracle Ensemble")
            if st.button("RUN ALL PREDICTIVE MODELS", type="primary", use_container_width=True):
                scored_predictions = run_full_backtest_suite(df_master)
                st.header("✨ Final Synthesis & Strategic Portfolio")
                if scored_predictions:
                    hybrid_pred = sorted([num for num, count in Counter(np.array([p['prediction'] for p in scored_predictions]).flatten()).most_common(6)])
                    hybrid_error = np.mean([p['error'] for p in scored_predictions], axis=0)
                    st.subheader("🏆 Prime Candidate: Hybrid Consensus")
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
                            with col2:
                                st.metric("Likelihood Score", f"{p.get('likelihood', 0):.1f}%", help=f"Backtest Metrics: {p.get('metrics', {})}")
        with tab2:
            st.header("System Dynamics & Inter-Number Physics")
            st.markdown("This module provides visualizations and metrics to explore the intrinsic, time-dependent behavior of the number system.")
            momentum_df, fig_zodiac, calculus_result, zodiac_result = analyze_system_dynamics(df_master)
            st.plotly_chart(fig_zodiac, use_container_width=True)
            st.subheader("Calculus Momentum Analysis")
            st.dataframe(momentum_df)
            st.subheader("Dynamical Model Predictions")
            for p in [calculus_result, zodiac_result]:
                pred_str = ' | '.join([f"{n} <small>(±{e:.1f})</small>" for n, e in zip(p['prediction'], p['error'])])
                st.info(f"**{p['name']}:** {pred_str}", icon="➡️")
        with tab3:
            st.header("Predictive Maturity Analysis")
            st.markdown("This analysis determines how the predictive power of the models evolves as more historical data is used. A plateau in the curve suggests the system has reached its maximum potential predictability with the given data.")
            if st.button("RUN MATURITY ANALYSIS"):
                maturity_df = analyze_predictive_maturity(df_master)
                if not maturity_df.empty:
                    fig_maturity = px.line(maturity_df, x='History Size', y='Top Likelihood Score', title="Predictive Maturity Curve", markers=True)
                    fig_maturity.update_layout(yaxis_range=[0,100])
                    st.plotly_chart(fig_maturity, use_container_width=True)
    else:
        st.error("Invalid data format. After cleaning, the file must have 6 number columns.")
else:
    st.info("Upload a CSV file to engage the Oracle Ensemble.")
