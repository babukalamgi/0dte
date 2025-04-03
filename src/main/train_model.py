import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds project root to sys.path

from utils.common_imports import *


def train_model():

    # Load dataset
    df = pd.read_csv("data/processed/spx_vix_data.csv")

    # Define feature and target columns
    feature_cols = ["year", "month", "day", "open_vix", "open"]
    target_cols = ["high", "low", "close"]

    # Extract features and targets
    features = df[feature_cols].copy()
    targets = df[target_cols].copy()

    # Scale data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets)

    joblib.dump(feature_scaler, "src/models/feature_scaler.pkl")   
    joblib.dump(target_scaler, "src/models/target_scaler.pkl")

    # Sequence creation
    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(window_size, len(X)):
            X_seq.append(X[i - window_size:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    window_size = 10
    X, y = create_sequences(features_scaled, targets_scaled, window_size)

    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        LSTM(32, activation='relu', return_sequences=False),  # return_sequences=False here
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3)
    ])


    # Compile the model
    model.compile(optimizer='adam', loss=MeanSquaredError())

    # Save best model
    checkpoint_path = "src/models/0dte_spx_vix_model.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, mode='min', verbose=1)

    # Train model
    model.fit(X, y, epochs=50, batch_size=32, callbacks=[checkpoint], verbose=1)

    return None