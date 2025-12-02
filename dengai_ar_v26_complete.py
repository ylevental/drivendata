"""
DengAI: Complete pipeline to generate submission_ar_v26.csv
Score: 23.0481 (Top 10%)

This script creates the final submission by:
1. Building base models with feature engineering
2. Creating an autoregressive model with iterative prediction
3. Blending them optimally (26% AR + 74% base blend)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DengAI: Generating submission_ar_v26.csv")
print("="*60)

# =============================================================================
# LOAD DATA
# =============================================================================
train_features = pd.read_csv('/mnt/user-data/uploads/dengue_features_train.csv')
train_labels = pd.read_csv('/mnt/user-data/uploads/dengue_labels_train.csv')
test_features = pd.read_csv('/mnt/user-data/uploads/dengue_features_test.csv')
submission = pd.read_csv('/mnt/user-data/uploads/submission_format.csv')

train = train_features.merge(train_labels, on=['city', 'year', 'weekofyear'])

print(f"Train: {len(train)}, Test: {len(test_features)}")

# =============================================================================
# PART 1: CREATE BASE BLEND (i12)
# This combines multiple model predictions
# =============================================================================
print("\n" + "="*60)
print("PART 1: Building base models for blending")
print("="*60)

def create_base_features_simple(df):
    """Simple feature engineering for base models."""
    df = df.copy()
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Seasonal
    df['sin_week'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['weekofyear'] / 52)
    
    # Key climate features with lags
    key_cols = ['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k',
                'station_avg_temp_c', 'reanalysis_min_air_temp_k', 'precipitation_amt_mm']
    
    for col in key_cols:
        for lag in [1, 2, 4, 8]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_roll4'] = df[col].rolling(4, min_periods=1).mean()
    
    # NDVI
    for col in ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']:
        df[f'{col}_lag4'] = df[col].shift(4)
    
    df = df.fillna(method='bfill').fillna(0)
    return df

def create_enhanced_features(df):
    """Enhanced feature engineering for improved model."""
    df = df.copy()
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Seasonal - multiple harmonics
    for k in [1, 2]:
        df[f'sin_{k}'] = np.sin(2 * np.pi * k * df['weekofyear'] / 52)
        df[f'cos_{k}'] = np.cos(2 * np.pi * k * df['weekofyear'] / 52)
    
    # All key climate columns
    humidity_cols = ['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k',
                     'reanalysis_relative_humidity_percent']
    temp_cols = ['station_avg_temp_c', 'station_min_temp_c', 'reanalysis_min_air_temp_k']
    precip_cols = ['precipitation_amt_mm', 'reanalysis_precip_amt_kg_per_m2']
    
    all_key_cols = humidity_cols + temp_cols + precip_cols
    
    for col in all_key_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 4, 6, 8, 10, 12]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            for w in [4, 8, 12]:
                df[f'{col}_rmean{w}'] = df[col].rolling(w, min_periods=1).mean()
    
    # NDVI
    for col in ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']:
        if col in df.columns:
            df[f'{col}_lag4'] = df[col].shift(4)
            df[f'{col}_lag8'] = df[col].shift(8)
    
    # Interactions
    if 'station_avg_temp_c' in df.columns and 'reanalysis_relative_humidity_percent' in df.columns:
        df['temp_x_humid'] = df['station_avg_temp_c'] * df['reanalysis_relative_humidity_percent']
    
    if 'station_avg_temp_c' in df.columns:
        df['temp_optimal'] = ((df['station_avg_temp_c'] >= 25) & 
                              (df['station_avg_temp_c'] <= 30)).astype(int)
    
    if 'precipitation_amt_mm' in df.columns:
        df['precip_cumsum4'] = df['precipitation_amt_mm'].rolling(4, min_periods=1).sum()
        df['precip_cumsum8'] = df['precipitation_amt_mm'].rolling(8, min_periods=1).sum()
    
    df = df.fillna(method='bfill').fillna(0)
    return df

exclude_cols = ['city', 'year', 'weekofyear', 'week_start_date', 'total_cases']

# --- Build "seasonal" baseline ---
print("\nBuilding seasonal baseline...")
seasonal_preds = {}
for city in ['sj', 'iq']:
    train_city = train[train['city'] == city]
    test_city = test_features[test_features['city'] == city]
    weekly_avg = train_city.groupby('weekofyear')['total_cases'].mean()
    preds = test_city['weekofyear'].map(weekly_avg).fillna(weekly_avg.mean())
    seasonal_preds[city] = np.round(preds).astype(int).values
seasonal = np.concatenate([seasonal_preds['sj'], seasonal_preds['iq']])
print(f"  Seasonal: mean={seasonal.mean():.2f}")

# --- Build "v2" model (GradientBoosting ensemble) ---
print("\nBuilding v2 model...")
v2_preds = {}
for city in ['sj', 'iq']:
    train_city = train[train['city'] == city].copy().reset_index(drop=True)
    test_city = test_features[test_features['city'] == city].copy().reset_index(drop=True)
    
    n_train = len(train_city)
    combined = pd.concat([train_city, test_city], ignore_index=True)
    combined = create_base_features_simple(combined)
    
    train_proc = combined.iloc[:n_train]
    test_proc = combined.iloc[n_train:]
    
    feature_cols = [c for c in train_proc.columns if c not in exclude_cols]
    
    X_train = train_proc[feature_cols].iloc[8:]
    y_train = train_proc['total_cases'].iloc[8:]
    X_test = test_proc[feature_cols]
    
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                                   min_samples_leaf=5, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    v2_preds[city] = np.clip(np.round(gb.predict(X_test)), 0, None).astype(int)
v2 = np.concatenate([v2_preds['sj'], v2_preds['iq']])
print(f"  V2: mean={v2.mean():.2f}")

# --- Build "ensemble" model ---
print("\nBuilding ensemble model...")
ensemble_preds = {}
for city in ['sj', 'iq']:
    train_city = train[train['city'] == city].copy().reset_index(drop=True)
    test_city = test_features[test_features['city'] == city].copy().reset_index(drop=True)
    
    n_train = len(train_city)
    combined = pd.concat([train_city, test_city], ignore_index=True)
    combined = create_base_features_simple(combined)
    
    train_proc = combined.iloc[:n_train]
    test_proc = combined.iloc[n_train:]
    
    feature_cols = [c for c in train_proc.columns if c not in exclude_cols]
    
    X_train = train_proc[feature_cols].iloc[8:]
    y_train = train_proc['total_cases'].iloc[8:]
    X_test = test_proc[feature_cols]
    
    # Multiple models
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.03,
                                   min_samples_leaf=5, subsample=0.8, random_state=42)
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)
    
    gb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    pred = (gb.predict(X_test) + rf.predict(X_test)) / 2
    ensemble_preds[city] = np.clip(np.round(pred), 0, None).astype(int)
ensemble = np.concatenate([ensemble_preds['sj'], ensemble_preds['iq']])
print(f"  Ensemble: mean={ensemble.mean():.2f}")

# --- Build "refined" model ---
print("\nBuilding refined model...")
refined_preds = {}
for city in ['sj', 'iq']:
    train_city = train[train['city'] == city].copy().reset_index(drop=True)
    test_city = test_features[test_features['city'] == city].copy().reset_index(drop=True)
    
    n_train = len(train_city)
    combined = pd.concat([train_city, test_city], ignore_index=True)
    combined = create_base_features_simple(combined)
    
    train_proc = combined.iloc[:n_train]
    test_proc = combined.iloc[n_train:]
    
    feature_cols = [c for c in train_proc.columns if c not in exclude_cols]
    
    X_train = train_proc[feature_cols].iloc[8:]
    y_train = train_proc['total_cases'].iloc[8:]
    X_test = test_proc[feature_cols]
    
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.03,
                                   min_samples_leaf=4, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    refined_preds[city] = np.clip(np.round(gb.predict(X_test)), 0, None).astype(int)
refined = np.concatenate([refined_preds['sj'], refined_preds['iq']])
print(f"  Refined: mean={refined.mean():.2f}")

# --- Build "improved" model (enhanced features) ---
print("\nBuilding improved model...")
improved_preds = {}
for city in ['sj', 'iq']:
    train_city = train[train['city'] == city].copy().reset_index(drop=True)
    test_city = test_features[test_features['city'] == city].copy().reset_index(drop=True)
    
    n_train = len(train_city)
    combined = pd.concat([train_city, test_city], ignore_index=True)
    combined = create_enhanced_features(combined)
    
    train_proc = combined.iloc[:n_train]
    test_proc = combined.iloc[n_train:]
    
    feature_cols = [c for c in train_proc.columns if c not in exclude_cols]
    
    skip = 12
    X_train = train_proc[feature_cols].iloc[skip:]
    y_train = train_proc['total_cases'].iloc[skip:]
    X_test = test_proc[feature_cols]
    
    # City-specific tuning
    if city == 'sj':
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                                       min_samples_leaf=4, subsample=0.8, random_state=42)
    else:
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=5, subsample=0.8, random_state=42)
    
    gb.fit(X_train, y_train)
    
    gb_log = GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.04,
                                       min_samples_leaf=5, subsample=0.8, random_state=42)
    gb_log.fit(X_train, np.log1p(y_train))
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    pred = (gb.predict(X_test) + np.expm1(gb_log.predict(X_test)) + rf.predict(X_test)) / 3
    improved_preds[city] = np.clip(np.round(pred), 0, None).astype(int)
improved = np.concatenate([improved_preds['sj'], improved_preds['iq']])
print(f"  Improved: mean={improved.mean():.2f}")

# --- Create f3 blend (base for i12) ---
f3 = 0.25 * seasonal + 0.25 * v2 + 0.25 * ensemble + 0.25 * refined
f3 = np.round(f3).astype(int)
print(f"\nF3 blend: mean={f3.mean():.2f}")

# --- Create i12 blend (12% improved + 88% f3) ---
i12 = 0.12 * improved + 0.88 * f3
i12 = np.round(i12).astype(int)
print(f"I12 blend: mean={i12.mean():.2f}")

# =============================================================================
# PART 2: CREATE AUTOREGRESSIVE MODEL
# =============================================================================
print("\n" + "="*60)
print("PART 2: Building autoregressive model")
print("="*60)

def create_ar_features(df):
    """Create features for autoregressive model."""
    df = df.copy()
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Seasonal
    df['sin_week'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['weekofyear'] / 52)
    df['sin_week2'] = np.sin(4 * np.pi * df['weekofyear'] / 52)
    df['cos_week2'] = np.cos(4 * np.pi * df['weekofyear'] / 52)
    
    # Climate features with lags
    key_cols = ['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k',
                'station_avg_temp_c', 'reanalysis_min_air_temp_k', 'precipitation_amt_mm']
    
    for col in key_cols:
        for lag in [1, 2, 3, 4, 8]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_roll4'] = df[col].rolling(4, min_periods=1).mean()
        df[f'{col}_roll8'] = df[col].rolling(8, min_periods=1).mean()
    
    # NDVI
    for col in ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']:
        df[f'{col}_lag4'] = df[col].shift(4)
    
    df = df.fillna(method='bfill').fillna(0)
    return df

def add_case_features(df, cases_series):
    """Add autoregressive case features."""
    df = df.copy()
    for lag in [1, 2, 3, 4]:
        df[f'cases_lag{lag}'] = cases_series.shift(lag)
    df['cases_roll4'] = cases_series.rolling(4, min_periods=1).mean()
    df['cases_roll8'] = cases_series.rolling(8, min_periods=1).mean()
    return df

ar_predictions = {}

for city in ['sj', 'iq']:
    print(f"\n--- {city.upper()} ---")
    
    train_city = train[train['city'] == city].copy().reset_index(drop=True)
    test_city = test_features[test_features['city'] == city].copy().reset_index(drop=True)
    
    # Create base features
    train_city = create_ar_features(train_city)
    test_city = create_ar_features(test_city)
    
    # Add case features for training
    train_city = add_case_features(train_city, train_city['total_cases'])
    
    # Feature columns (including case lags)
    feature_cols = [c for c in train_city.columns if c not in exclude_cols]
    
    # Train on data after initial lags
    skip = 8
    X_train = train_city[feature_cols].iloc[skip:]
    y_train = train_city['total_cases'].iloc[skip:]
    
    print(f"Features: {len(feature_cols)}, Training samples: {len(X_train)}")
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=250, max_depth=5, learning_rate=0.04,
        min_samples_leaf=4, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    
    train_mae = mean_absolute_error(y_train, np.round(model.predict(X_train)))
    print(f"Train MAE: {train_mae:.2f}")
    
    # ITERATIVE PREDICTION for test set
    last_cases = list(train_city['total_cases'].iloc[-8:].values)
    test_preds = []
    
    for i in range(len(test_city)):
        row = test_city.iloc[[i]].copy()
        
        all_cases = last_cases + test_preds
        cases_series = pd.Series(all_cases)
        
        for lag in [1, 2, 3, 4]:
            row[f'cases_lag{lag}'] = cases_series.iloc[-lag] if len(cases_series) >= lag else 0
        
        row['cases_roll4'] = cases_series.iloc[-4:].mean() if len(cases_series) >= 4 else cases_series.mean()
        row['cases_roll8'] = cases_series.iloc[-8:].mean() if len(cases_series) >= 8 else cases_series.mean()
        
        X_row = row[feature_cols]
        pred = max(0, round(model.predict(X_row)[0]))
        test_preds.append(pred)
    
    ar_predictions[city] = np.array(test_preds).astype(int)
    print(f"Predictions: range [{min(test_preds)}, {max(test_preds)}], mean={np.mean(test_preds):.1f}")

ar = np.concatenate([ar_predictions['sj'], ar_predictions['iq']])
print(f"\nAutoregressive model: mean={ar.mean():.2f}")

# =============================================================================
# PART 3: CREATE FINAL BLEND (26% AR + 74% i12)
# =============================================================================
print("\n" + "="*60)
print("PART 3: Creating final blend")
print("="*60)

ar_weight = 0.26
final_blend = ar_weight * ar + (1 - ar_weight) * i12
final_blend = np.round(final_blend).astype(int)

print(f"AR weight: {ar_weight}")
print(f"Final blend: mean={final_blend.mean():.2f}")

# Create submission
final_submission = submission.copy()
final_submission['total_cases'] = final_blend
final_submission.to_csv('submission_ar_v26_standalone.csv', index=False)

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"Saved: submission_ar_v26_standalone.csv")
print(f"Mean: {final_blend.mean():.2f}")
print(f"Min: {final_blend.min()}, Max: {final_blend.max()}")
print("\nSubmission preview:")
print(final_submission.head(10))
