# =================================================================================================
# LottoSphere v14.0: The Pattern Forecaster
#
# AUTHOR: Subject Matter Expert AI (Complex Systems, Mathematics & AI/ML)
# DATE: 2024-07-25
# VERSION: 14.0 (Probabilistic Number Forecasting)
#
# DESCRIPTION:
# This definitive version completes the analytical pipeline by translating the predicted system
# state (pattern) into a probabilistic forecast for the individual numbers. It introduces a
# rigorous walk-forward backtesting framework to scientifically measure the historical accuracy
# of the entire pattern-to-number forecasting process.
#
# CORE METHODOLOGY:
# 1. PATTERN IDENTIFICATION: Transforms draws into high-dimensional pattern vectors and uses
#    HDBSCAN to identify recurring system states (clusters).
# 2. STATE TRANSITION MODELING: A Markov Chain predicts the most likely next state.
# 3. PROBABILISTIC NUMBER FORECASTING (NEW): For each predicted state, the engine retrieves the
#    historical probability distribution for each of the six number slots and generates a
#    forecast with a data-driven Likelihood Score for each number.
# 4. WALK-FORWARD BACKTESTING (NEW): The entire two-stage process is rigorously backtested on
#    a validation set to produce a final, honest Historical Forecast Accuracy score.
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
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

# --- 1. APPLICATION CONFIGURATION ---
st.set_page_config(
    page_title="LottoSphere v14.0: Pattern Forecaster",
    page_icon="🎯",
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
    if df.shape[1] > 6: df = df.iloc[:, :6]
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

def get_digital_root(n):
    return (n - 1) % 9 + 1 if n > 0 else 0

@st.cache_data
def create_pattern_dataframe(_df):
    patterns = pd.DataFrame(index=_df.index)
    df_nums = _df.iloc[:, :6]
    df_sorted = pd.DataFrame(np.sort(df_nums.values, axis=1), index=_df.index)
    
    patterns['sum'] = df_nums.sum(axis=1)
    patterns['std'] = df_nums.std(axis=1)
    patterns['odd_count'] = df_nums.apply(lambda r: sum(n % 2 for n in r), axis=1)
    patterns['prime_count'] = df_nums.apply(lambda r: sum(is_prime(n) for n in r), axis=1)
    gaps = df_sorted.diff(axis=1).dropna(axis=1)
    patterns['mean_gap'] = gaps.mean(axis=1)
    
    max_num = df_nums.values.max()
    low_b, high_b = max_num / 3, 2 * max_num / 3
    patterns['low_num_count'] = df_nums.apply(lambda r: sum(1 for n in r if n <= low_b), axis=1)
    patterns['mid_num_count'] = df_nums.apply(lambda r: sum(1 for n in r if low_b < n <= high_b), axis=1)
    
    return patterns

@st.cache_data
def find_system_states(_pattern_df):
    scaler = StandardScaler()
    pattern_scaled = scaler.fit_transform(_pattern_df)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=2, random_state=42)
    embedding = reducer.fit_transform(pattern_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)
    return cluster_labels, embedding

@st.cache_data
def build_markov_transition_matrix(cluster_labels):
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    matrix = np.zeros((n_clusters, n_clusters))
    for i in range(len(cluster_labels) - 1):
        current_state, next_state = cluster_labels[i], cluster_labels[i+1]
        if current_state != -1 and next_state != -1:
            matrix[current_state, next_state] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums

@st.cache_data
def calculate_cluster_distributions(_df, cluster_labels):
    """Pre-calculates the probability distribution of numbers for each position within each cluster."""
    df_sorted = pd.DataFrame(np.sort(_df.iloc[:,:6].values, axis=1), index=_df.index, columns=[f'Pos_{i+1}' for i in range(6)])
    df_sorted['cluster'] = cluster_labels
    
    distributions = {}
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    for i in range(n_clusters):
        cluster_df = df_sorted[df_sorted['cluster'] == i]
        if not cluster_df.empty:
            cluster_dist = {}
            for pos in range(6):
                counts = cluster_df.iloc[:, pos].value_counts(normalize=True)
                cluster_dist[f'Pos_{pos+1}'] = counts
            distributions[i] = cluster_dist
            
    return distributions
# --- 5. PREDICTION & BACKTESTING FUNCTIONS ---

def generate_prediction_from_state(target_state, cluster_dists):
    """Generates a 6-number prediction and likelihood scores from a predicted state."""
    if target_state not in cluster_dists:
        return [0]*6, [0]*6 # Return empty if state has no historical data

    prediction = []
    likelihoods = []
    for pos in range(6):
        pos_dist = cluster_dists[target_state].get(f'Pos_{pos+1}')
        if pos_dist is not None and not pos_dist.empty:
            most_likely_num = pos_dist.idxmax()
            likelihood = pos_dist.max()
            prediction.append(most_likely_num)
            likelihoods.append(likelihood)
        else:
            prediction.append(0) # Placeholder
            likelihoods.append(0)
    
    return prediction, likelihoods

@st.cache_data
def run_walk_forward_backtest(df):
    """Performs a rigorous walk-forward validation of the entire pipeline."""
    split_point = int(len(df) * 0.8)
    val_df = df.iloc[split_point:]
    
    y_preds, y_trues = [], []
    
    progress_bar = st.progress(0, text="Performing walk-forward validation...")
    for i in range(len(val_df) - 1):
        # Train on all data up to the point of prediction
        historical_df = df.iloc[:split_point + i + 1]
        
        # 1. Create patterns and find states on historical data
        pattern_df = create_pattern_dataframe(historical_df)
        cluster_labels, _ = find_system_states(pattern_df)
        
        # 2. Build transition matrix and get last state
        transition_matrix = build_markov_transition_matrix(cluster_labels)
        last_state = cluster_labels[-1]
        
        if last_state != -1:
            # 3. Predict next state
            predicted_next_state = np.argmax(transition_matrix[last_state])
            
            # 4. Calculate cluster distributions and generate number prediction
            cluster_dists = calculate_cluster_distributions(historical_df, cluster_labels)
            prediction, _ = generate_prediction_from_state(predicted_next_state, cluster_dists)
            y_preds.append(prediction)
            y_trues.append(val_df.iloc[i + 1, :6].tolist())
        
        progress_bar.progress((i + 1) / (len(val_df) - 1), text=f"Validating Draw {split_point + i + 1}...")
    
    progress_bar.empty()
    if not y_preds: return {}

    # Calculate overall performance metrics
    hits = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues, y_preds))
    precise_hits = sum(1 for yt, yp in zip(y_trues, y_preds) if len(set(yt) & set(yp)) >= 3)
    accuracy = hits / len(y_trues)
    precision = precise_hits / len(y_trues)
    
    return {'Avg Hits per Draw': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.2%}"}

# =================================================================================================
# Main Application UI & Logic
# =================================================================================================

st.title("🎯 LottoSphere v14.0: The Pattern Forecaster")
st.markdown("This engine predicts the **underlying mathematical pattern** of the next draw and then generates a **probabilistic number forecast** based on that pattern. The entire methodology is rigorously backtested to assess its historical forecasting accuracy.")

if 'data_warning' not in st.session_state: st.session_state.data_warning = None
uploaded_file = st.sidebar.file_uploader("Upload Number.csv", type=["csv"])

if uploaded_file:
    df_master = load_data(uploaded_file)
    if st.session_state.data_warning: st.warning(st.session_state.data_warning); st.session_state.data_warning = None

    if df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        
        if st.sidebar.button("FORECAST NEXT PATTERN & NUMBERS", type="primary", use_container_width=True):
            
            # --- STAGE 1 & 2: Pattern Analysis and State Identification ---
            st.header("Stage 1: Identifying System States")
            st.markdown("The engine first transforms each draw into a mathematical pattern and then uses advanced clustering to identify recurring states in the system's history.")
            with st.spinner("Analyzing patterns and identifying system states..."):
                pattern_df = create_pattern_dataframe(df_master)
                cluster_labels, embedding = find_system_states(pattern_df)
                pattern_df['State'] = [f'State {l}' if l != -1 else 'Chaotic' for l in cluster_labels]
            
            embedding_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
            embedding_df['State'] = pattern_df['State']
            fig = px.scatter(embedding_df, x='UMAP 1', y='UMAP 2', color='State', title="Topological Map of System States (UMAP)")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Identified **{len(set(cluster_labels)) - 1}** distinct behavioral states.")
            st.markdown("---")

            # --- STAGE 3: State Transition Modeling and Prediction ---
            st.header("Stage 2: Predicting the Next System State")
            st.markdown("Using a Markov Chain model, we analyze the historical flow between states to predict the most probable next state.")
            with st.spinner("Building State Transition Model..."):
                transition_matrix = build_markov_transition_matrix(cluster_labels)
            
            last_state = cluster_labels[-1]
            if last_state != -1:
                st.info(f"The system's most recent draw was identified as belonging to **State {last_state}**.")
                next_state_probs = transition_matrix[last_state]
                predicted_next_state = np.argmax(next_state_probs)
                likelihood = next_state_probs[predicted_next_state]
                st.success(f"The model predicts the system will transition to **State {predicted_next_state}** with a likelihood of **{likelihood:.1%}**.")
                
                # --- STAGE 4: Probabilistic Number Forecast ---
                st.header("Stage 3: Probabilistic Number Forecast")
                st.markdown(f"Based on the prediction that the next draw will be a **State {predicted_next_state}** pattern, the engine has calculated the most likely number for each of the six sorted positions.")
                with st.spinner("Calculating cluster distributions and generating forecast..."):
                    cluster_dists = calculate_cluster_distributions(df_master, cluster_labels)
                    prediction, likelihoods = generate_prediction_from_state(predicted_next_state, cluster_dists)
                
                if prediction[0] != 0:
                    pred_df = pd.DataFrame({
                        'Position': [f'Position {i+1}' for i in range(6)],
                        'Predicted Number': prediction,
                        'Likelihood Score': [f"{l:.1%}" for l in likelihoods]
                    })
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)

                    st.subheader("🏆 Prime Candidate Set")
                    st.success(f"## `{sorted(prediction)}`")
                else:
                    st.error("The predicted state has no historical precedent, cannot generate a number forecast.")
            else:
                st.warning("The system's most recent draw was a chaotic transition. A specific state cannot be predicted.")

            # --- STAGE 5: Backtesting and Performance Assessment ---
            st.header("Stage 4: Historical Forecasting Accuracy")
            st.markdown("To scientifically validate this entire two-stage methodology, we performed a rigorous walk-forward backtest on the last 20% of the historical data. The results below show how accurately this pattern-based approach would have predicted the numbers in the past.")
            backtest_results = run_walk_forward_backtest(df_master)
            if backtest_results:
                st.metric("Historical Average Hits per Draw", backtest_results['Avg Hits per Draw'])
                st.metric("Historical High-Tier (3+) Hit Rate", backtest_results['3+ Hit Rate'])
            else:
                st.warning("Not enough data to perform a full backtest.")

    else:
        st.error(f"Invalid data format. After cleaning, the file must have 6 number columns.")
else:
    st.info("Upload a CSV file to engage the Pattern Forecaster Engine.")
