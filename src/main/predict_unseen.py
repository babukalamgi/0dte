import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds project root to sys.path

from utils.common_imports import *

def predict_unseen_data():

    # Load unseen data
    unseen_df = pd.read_csv("data/live_market_data/spx_vix_live_market.csv")

    # Define columns
    feature_cols = ["year", "month", "day", "open_vix", "open"]
    target_cols = ["high", "low", "close"]
    window_size = 10

    # Clean invalid values
    unseen_df = unseen_df.replace([np.inf, -np.inf], np.nan)
    clean_unseen_df = unseen_df.dropna(subset=feature_cols).reset_index(drop=True)

    # Load scalers
    feature_scaler = joblib.load("src/models/feature_scaler.pkl")
    target_scaler = joblib.load("src/models/target_scaler.pkl")

    # Scale input features
    features_scaled = feature_scaler.transform(clean_unseen_df[feature_cols])

    # Create input sequences — one for each row, padded if needed
    def create_full_sequences(data, window_size):
        sequences = []
        for i in range(len(data)):
            if i < window_size - 1:
                pad = np.tile(data[0], (window_size - 1 - i, 1))
                seq = np.vstack([pad, data[:i + 1]])
            else:
                seq = data[i - window_size + 1:i + 1]
            sequences.append(seq)
        return np.array(sequences)

    X_input = create_full_sequences(features_scaled, window_size)


    # Load the saved model and predict
    model = load_model("src/models/0dte_spx_vix_model.h5")
    pred_scaled = model.predict(X_input)
    pred = target_scaler.inverse_transform(pred_scaled)

    # Prepare results
    results = pd.DataFrame(pred, columns=[f"pred_{col}" for col in target_cols])

    if all(col in clean_unseen_df.columns for col in target_cols):
        actual = clean_unseen_df[target_cols].iloc[-len(pred):].reset_index(drop=True)
        for col in target_cols:
            results[f"actual_{col}"] = actual[col]

        # Error metrics
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        print(f" MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    else:
        print("Actual values not available for error metrics.")

    # Show predictions
    results['high_diff'] = results['pred_high'] - results['actual_high']
    results['low_diff'] = results['pred_low'] - results['actual_low']
    results['close_diff'] = results['pred_close'] - results['actual_close']
    results[['actual_high', 'pred_high', 'high_diff',  'actual_low', 'pred_low', 'low_diff', 'actual_close', 'pred_close', 'close_diff']]

    return None