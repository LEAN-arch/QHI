# --- 5. BACKTESTING & SCORING (Temporally Rigorous)---
@st.cache_data
def backtest_and_score(df):
    st.markdown("---")
    st.subheader("Stage 1: Rigorous Walk-Forward Backtesting")
    st.markdown("To accurately measure historical performance, we simulate predicting each draw using **only the data that came before it**. This is computationally intensive but provides a true measure of each model's forecasting ability.")
    
    split_point = int(len(df) * 0.8)
    val_df = df.iloc[split_point:]
    
    # --- Walk-forward validation for non-ML models ---
    model_funcs = {
        "Quantum Fluctuation": analyze_quantum_fluctuations,
        "Stochastic Resonance": analyze_stochastic_resonance,
        "Bayesian GMM Inference": analyze_gmm_inference,
        "Topological AI (UMAP+HDBSCAN)": analyze_topological_ai
    }
    
    y_preds_map = {name: [] for name in model_funcs}
    y_trues = val_df.iloc[1:, :6].values.tolist()
    
    progress_bar = st.progress(0, text="Backtesting Acausal & Stochastic Models...")
    total_steps = len(val_df) - 1
    for i in range(total_steps):
        historical_df = df.iloc[:split_point + i + 1]
        for name, func in model_funcs.items():
            y_preds_map[name].append(func(historical_df)['prediction'])
        progress_bar.progress((i + 1) / total_steps, text=f"Backtesting Draw {i+1}/{total_steps}")
    progress_bar.empty()

    scored_predictions = []
    for name, y_preds in y_preds_map.items():
        if not y_preds: continue
        
        hits = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues, y_preds))
        precise_hits = sum(1 for yt, yp in zip(y_trues, y_preds) if len(set(yt) & set(yp)) >= 3)
        
        accuracy = hits / len(y_trues)
        precision = precise_hits / len(y_trues)
        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        
        acc_score = min(100, (accuracy / 1.2) * 100)
        prec_score = min(100, (precision / 0.1) * 100)
        rmse_score = max(0, 100 - (rmse / 20.0) * 100)
        likelihood = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
        
        final_pred_obj = model_funcs[name](df) # Get final prediction on full data
        final_pred_obj['likelihood'] = likelihood
        final_pred_obj['metrics'] = {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
        scored_predictions.append(final_pred_obj)
            
    # --- Backtesting for the Ensemble AI model ---
    with st.spinner("Backtesting Ensemble AI Model (LightGBM)..."):
        ensemble_models = train_ensemble_models(df)
        ensemble_pred_final = predict_with_ensemble(df, ensemble_models)
        
        features_full = feature_engineering(df)
        y_true_full = df.shift(-1).dropna().iloc[:, :6]
        common_index = features_full.index.intersection(y_true_full.index)
        features_aligned = features_full.loc[common_index]
        y_true_aligned = y_true_full.loc[common_index]

        _, X_test, _, y_test = train_test_split(features_aligned, y_true_aligned, test_size=0.2, shuffle=False)
        
        y_preds_ensemble = [sorted(np.round([m.predict(X_test.iloc[i:i+1])[0] for m in ensemble_models['median']]).astype(int)) for i in range(len(X_test))]
        y_trues_ensemble = y_test.values.tolist()
    
        if y_trues_ensemble:
            accuracy = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues_ensemble, y_preds_ensemble)) / len(y_trues_ensemble)
            precision = sum(1 for yt, yp in zip(y_trues_ensemble, y_preds_ensemble) if len(set(yt) & set(yp)) >= 3) / len(y_trues_ensemble)
            rmse = np.sqrt(mean_squared_error(y_trues_ensemble, y_preds_ensemble))
            
            acc_score = min(100, (accuracy / 1.2) * 100); prec_score = min(100, (precision / 0.1) * 100); rmse_score = max(0, 100 - (rmse / 20.0) * 100)
            ensemble_pred_final['likelihood'] = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
            ensemble_pred_final['metrics'] = {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
        else:
            ensemble_pred_final['likelihood'] = 0; ensemble_pred_final['metrics'] = {'Avg Hits': "N/A", '3+ Hit Rate': "N/A", 'RMSE': "N/A"}
        scored_predictions.append(ensemble_pred_final)

    return sorted(scored_predictions, key=lambda x: x['likelihood'], reverse=True)

# =================================================================================================
# Main Application UI & Logic
# =================================================================================================

st.title("💠 LottoSphere X: The Oracle Ensemble")
st.markdown("An advanced instrument for modeling complex systems. This engine runs two parallel suites of analyses—**Acausal Physics** and **Stochastic AI**—to identify candidate sets with the highest likelihood based on rigorous, time-series backtesting.")

uploaded_file = st.sidebar.file_uploader("Upload Number.csv", type=["csv"])

if uploaded_file:
    df_master = load_data(uploaded_file)
    if df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        
        if st.sidebar.button("💠 ENGAGE ORACLE ENSEMBLE", type="primary", use_container_width=True):
            
            scored_predictions = backtest_and_score(df_master)
            
            st.header("✨ Final Synthesis & Strategic Portfolio")
            st.markdown("The Oracle has completed all analyses. Below is the final consensus and the ranked predictions from each model, complete with quantified uncertainty and a **Likelihood Score** based on historical forecasting performance.")
            
            if scored_predictions:
                # Create Hybrid Consensus Prediction
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
# --- 5. BACKTESTING & SCORING (Temporally Rigorous)---
@st.cache_data
def backtest_and_score(df):
    st.markdown("---")
    st.subheader("Stage 1: Rigorous Walk-Forward Backtesting")
    st.markdown("To accurately measure historical performance, we simulate predicting each draw using **only the data that came before it**. This is computationally intensive but provides a true measure of each model's forecasting ability.")
    
    split_point = int(len(df) * 0.8)
    val_df = df.iloc[split_point:]
    
    # --- Walk-forward validation for non-ML models ---
    model_funcs = {
        "Quantum Fluctuation": analyze_quantum_fluctuations,
        "Stochastic Resonance": analyze_stochastic_resonance,
        "Bayesian GMM Inference": analyze_gmm_inference,
        "Topological AI (UMAP+HDBSCAN)": analyze_topological_ai
    }
    
    y_preds_map = {name: [] for name in model_funcs}
    y_trues = val_df.iloc[1:, :6].values.tolist()
    
    progress_bar = st.progress(0, text="Backtesting Acausal & Stochastic Models...")
    total_steps = len(val_df) - 1
    for i in range(total_steps):
        historical_df = df.iloc[:split_point + i + 1]
        for name, func in model_funcs.items():
            y_preds_map[name].append(func(historical_df)['prediction'])
        progress_bar.progress((i + 1) / total_steps, text=f"Backtesting Draw {i+1}/{total_steps}")
    progress_bar.empty()

    scored_predictions = []
    for name, y_preds in y_preds_map.items():
        if not y_preds: continue
        
        hits = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues, y_preds))
        precise_hits = sum(1 for yt, yp in zip(y_trues, y_preds) if len(set(yt) & set(yp)) >= 3)
        
        accuracy = hits / len(y_trues)
        precision = precise_hits / len(y_trues)
        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        
        acc_score = min(100, (accuracy / 1.2) * 100)
        prec_score = min(100, (precision / 0.1) * 100)
        rmse_score = max(0, 100 - (rmse / 20.0) * 100)
        likelihood = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
        
        final_pred_obj = model_funcs[name](df) # Get final prediction on full data
        final_pred_obj['likelihood'] = likelihood
        final_pred_obj['metrics'] = {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
        scored_predictions.append(final_pred_obj)
            
    # --- Backtesting for the Ensemble AI model ---
    with st.spinner("Backtesting Ensemble AI Model (LightGBM)..."):
        ensemble_models = train_ensemble_models(df)
        ensemble_pred_final = predict_with_ensemble(df, ensemble_models)
        
        features_full = feature_engineering(df)
        y_true_full = df.shift(-1).dropna().iloc[:, :6]
        common_index = features_full.index.intersection(y_true_full.index)
        features_aligned = features_full.loc[common_index]
        y_true_aligned = y_true_full.loc[common_index]

        _, X_test, _, y_test = train_test_split(features_aligned, y_true_aligned, test_size=0.2, shuffle=False)
        
        y_preds_ensemble = [sorted(np.round([m.predict(X_test.iloc[i:i+1])[0] for m in ensemble_models['median']]).astype(int)) for i in range(len(X_test))]
        y_trues_ensemble = y_test.values.tolist()
    
        if y_trues_ensemble:
            accuracy = sum(len(set(yt) & set(yp)) for yt, yp in zip(y_trues_ensemble, y_preds_ensemble)) / len(y_trues_ensemble)
            precision = sum(1 for yt, yp in zip(y_trues_ensemble, y_preds_ensemble) if len(set(yt) & set(yp)) >= 3) / len(y_trues_ensemble)
            rmse = np.sqrt(mean_squared_error(y_trues_ensemble, y_preds_ensemble))
            
            acc_score = min(100, (accuracy / 1.2) * 100); prec_score = min(100, (precision / 0.1) * 100); rmse_score = max(0, 100 - (rmse / 20.0) * 100)
            ensemble_pred_final['likelihood'] = 0.5 * acc_score + 0.3 * prec_score + 0.2 * rmse_score
            ensemble_pred_final['metrics'] = {'Avg Hits': f"{accuracy:.2f}", '3+ Hit Rate': f"{precision:.1%}", 'RMSE': f"{rmse:.2f}"}
        else:
            ensemble_pred_final['likelihood'] = 0; ensemble_pred_final['metrics'] = {'Avg Hits': "N/A", '3+ Hit Rate': "N/A", 'RMSE': "N/A"}
        scored_predictions.append(ensemble_pred_final)

    return sorted(scored_predictions, key=lambda x: x['likelihood'], reverse=True)

# =================================================================================================
# Main Application UI & Logic
# =================================================================================================

st.title("💠 LottoSphere X: The Oracle Ensemble")
st.markdown("An advanced instrument for modeling complex systems. This engine runs two parallel suites of analyses—**Acausal Physics** and **Stochastic AI**—to identify candidate sets with the highest likelihood based on rigorous, time-series backtesting.")

uploaded_file = st.sidebar.file_uploader("Upload Number.csv", type=["csv"])

if uploaded_file:
    df_master = load_data(uploaded_file)
    if df_master.shape[1] == 6:
        st.sidebar.success(f"Loaded and validated {len(df_master)} historical draws.")
        
        if st.sidebar.button("💠 ENGAGE ORACLE ENSEMBLE", type="primary", use_container_width=True):
            
            scored_predictions = backtest_and_score(df_master)
            
            st.header("✨ Final Synthesis & Strategic Portfolio")
            st.markdown("The Oracle has completed all analyses. Below is the final consensus and the ranked predictions from each model, complete with quantified uncertainty and a **Likelihood Score** based on historical forecasting performance.")
            
            if scored_predictions:
                # Create Hybrid Consensus Prediction
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
