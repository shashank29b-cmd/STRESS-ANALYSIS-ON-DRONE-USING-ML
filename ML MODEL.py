import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# Load the dataset
print("=" * 70)
print("ML STRESS PREDICTOR - TRAINING & HYPERPARAMETER TUNING")
print("=" * 70)

# Find the most recent CSV file
import glob
import os
csv_files = glob.glob('fea_dataset_*.csv')
if not csv_files:
    print("Error: No dataset file found. Please run the dataset generator first.")
    exit()

latest_file = max(csv_files, key=os.path.getctime)
print(f"\nLoading dataset: {latest_file}")
df = pd.read_csv(latest_file)
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Prepare features and target
print("\nPreparing features and target variable...")

# Encode categorical variables
label_encoder = LabelEncoder()
df['material_encoded'] = label_encoder.fit_transform(df['material_name'])

# Feature selection - exclude target variables and material name
feature_columns = [
    'length_mm', 'width_mm', 'height_mm', 'thickness_mm', 
    'area_mm2', 'moment_of_inertia_mm4',
    'youngs_modulus_MPa', 'poisson_ratio', 'yield_strength_MPa', 'density_kg_m3',
    'force_N', 'moment_Nmm', 'pressure_MPa', 'torque_Nmm',
    'load_type', 'boundary_condition', 'material_encoded'
]

X = df[feature_columns]
y = df['max_stress_MPa']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and encoder
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\n✓ Scaler and encoder saved")

# ============================================================================
# MODEL 1: Random Forest with Hyperparameter Tuning
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: RANDOM FOREST REGRESSOR")
print("=" * 70)

# Reduced parameter grid to avoid memory issues
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Starting GridSearch for Random Forest...")
print(f"Parameter grid: {rf_param_grid}")
print("This may take several minutes...\n")

start_time = time.time()
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=2),  # Reduced n_jobs
    rf_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=2,  # Reduced parallel jobs
    verbose=2
)
rf_grid.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"\n✓ Random Forest training complete!")
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV score: {rf_grid.best_score_:.4f}")
print(f"Training time: {rf_time:.2f} seconds")

# Evaluate Random Forest
rf_model = rf_grid.best_estimator_
y_pred_rf = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_accuracy = rf_r2 * 100

print(f"\nRandom Forest Performance:")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  Accuracy: {rf_accuracy:.2f}%")
print(f"  RMSE: {rf_rmse:.2f} MPa")
print(f"  MAE: {rf_mae:.2f} MPa")

# Test prediction time
start_pred = time.time()
_ = rf_model.predict(X_test[:100])
pred_time_rf = (time.time() - start_pred) / 100
print(f"  Avg prediction time: {pred_time_rf*1000:.2f} ms")

# Free memory
gc.collect()

# ============================================================================
# MODEL 2: Gradient Boosting with Hyperparameter Tuning
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: GRADIENT BOOSTING REGRESSOR")
print("=" * 70)

# Reduced parameter grid
gb_param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7],
    'subsample': [0.8, 1.0]
}

print("Starting GridSearch for Gradient Boosting...")
print(f"Parameter grid: {gb_param_grid}")
print("This may take several minutes...\n")

start_time = time.time()
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=2,  # Reduced parallel jobs
    verbose=2
)
gb_grid.fit(X_train, y_train)
gb_time = time.time() - start_time

print(f"\n✓ Gradient Boosting training complete!")
print(f"Best parameters: {gb_grid.best_params_}")
print(f"Best CV score: {gb_grid.best_score_:.4f}")
print(f"Training time: {gb_time:.2f} seconds")

# Evaluate Gradient Boosting
gb_model = gb_grid.best_estimator_
y_pred_gb = gb_model.predict(X_test)

gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_accuracy = gb_r2 * 100

print(f"\nGradient Boosting Performance:")
print(f"  R² Score: {gb_r2:.4f}")
print(f"  Accuracy: {gb_accuracy:.2f}%")
print(f"  RMSE: {gb_rmse:.2f} MPa")
print(f"  MAE: {gb_mae:.2f} MPa")

start_pred = time.time()
_ = gb_model.predict(X_test[:100])
pred_time_gb = (time.time() - start_pred) / 100
print(f"  Avg prediction time: {pred_time_gb*1000:.2f} ms")

# Free memory
gc.collect()

# ============================================================================
# MODEL 3: Neural Network (MLP)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 3: NEURAL NETWORK (MLP REGRESSOR)")
print("=" * 70)

# Simplified parameter grid
mlp_param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'activation': ['relu'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01]
}

print("Starting GridSearch for Neural Network...")
print(f"Parameter grid: {mlp_param_grid}")
print("This may take several minutes...\n")

start_time = time.time()
mlp_grid = GridSearchCV(
    MLPRegressor(max_iter=300, random_state=42, early_stopping=True),
    mlp_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=2,  # Reduced parallel jobs
    verbose=2
)
mlp_grid.fit(X_train_scaled, y_train)
mlp_time = time.time() - start_time

print(f"\n✓ Neural Network training complete!")
print(f"Best parameters: {mlp_grid.best_params_}")
print(f"Best CV score: {mlp_grid.best_score_:.4f}")
print(f"Training time: {mlp_time:.2f} seconds")

# Evaluate Neural Network
mlp_model = mlp_grid.best_estimator_
y_pred_mlp = mlp_model.predict(X_test_scaled)

mlp_r2 = r2_score(y_test, y_pred_mlp)
mlp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mlp_mae = mean_absolute_error(y_test, y_pred_mlp)
mlp_accuracy = mlp_r2 * 100

print(f"\nNeural Network Performance:")
print(f"  R² Score: {mlp_r2:.4f}")
print(f"  Accuracy: {mlp_accuracy:.2f}%")
print(f"  RMSE: {mlp_rmse:.2f} MPa")
print(f"  MAE: {mlp_mae:.2f} MPa")

start_pred = time.time()
_ = mlp_model.predict(X_test_scaled[:100])
pred_time_mlp = (time.time() - start_pred) / 100
print(f"  Avg prediction time: {pred_time_mlp*1000:.2f} ms")

# Free memory
gc.collect()

# ============================================================================
# COMPARE MODELS AND SELECT BEST
# ============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = {
    'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
    'R² Score': [rf_r2, gb_r2, mlp_r2],
    'Accuracy (%)': [rf_accuracy, gb_accuracy, mlp_accuracy],
    'RMSE (MPa)': [rf_rmse, gb_rmse, mlp_rmse],
    'MAE (MPa)': [rf_mae, gb_mae, mlp_mae],
    'Training Time (s)': [rf_time, gb_time, mlp_time],
    'Prediction Time (ms)': [pred_time_rf*1000, pred_time_gb*1000, pred_time_mlp*1000]
}

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# Select best model based on R² score
best_idx = np.argmax([rf_r2, gb_r2, mlp_r2])
best_models = [rf_model, gb_model, mlp_model]
best_model = best_models[best_idx]
best_model_name = results['Model'][best_idx]
best_use_scaled = (best_idx == 2)  # Neural Network uses scaled data

print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Best Accuracy: {results['Accuracy (%)'][best_idx]:.2f}%")

# Save best model
model_filename = f'stress_predictor_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
joblib.dump({
    'model': best_model,
    'model_name': best_model_name,
    'use_scaled': best_use_scaled,
    'feature_columns': feature_columns
}, model_filename)

print(f"✓ Best model saved as: {model_filename}")

# Save all results to CSV
results_filename = f'model_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
results_df.to_csv(results_filename, index=False)
print(f"✓ Performance metrics saved as: {results_filename}")

# Save detailed results to text file
txt_filename = f'model_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
with open(txt_filename, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ML STRESS PREDICTOR - TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {latest_file}\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("MODEL PERFORMANCE COMPARISON\n")
    f.write("=" * 70 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("=" * 70 + "\n")
    f.write(f"BEST MODEL: {best_model_name}\n")
    f.write("=" * 70 + "\n")
    f.write(f"Accuracy: {results['Accuracy (%)'][best_idx]:.2f}%\n")
    f.write(f"R² Score: {results['R² Score'][best_idx]:.4f}\n")
    f.write(f"RMSE: {results['RMSE (MPa)'][best_idx]:.2f} MPa\n")
    f.write(f"MAE: {results['MAE (MPa)'][best_idx]:.2f} MPa\n")
    f.write(f"Prediction Time: {results['Prediction Time (ms)'][best_idx]:.2f} ms\n")
    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write("BEST MODEL HYPERPARAMETERS\n")
    f.write("=" * 70 + "\n")
    if best_idx == 0:
        f.write(str(rf_grid.best_params_))
    elif best_idx == 1:
        f.write(str(gb_grid.best_params_))
    else:
        f.write(str(mlp_grid.best_params_))

print(f"✓ Detailed report saved as: {txt_filename}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nFiles generated:")
print(f"  • {model_filename} - Trained model")
print(f"  • {results_filename} - Performance metrics (CSV)")
print(f"  • {txt_filename} - Detailed report (TXT)")
print(f"  • scaler.pkl - Feature scaler")
print(f"  • label_encoder.pkl - Material encoder")
print("\nYou can now use these files with the GUI application!")
print("=" * 70)
