import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds project root to sys.path

from src.main.preprocessing import *
from src.main.train_model import *
from src.main.predict_unseen import *
from src.main.plot_predictions import *

def main():

    # === Step 1: Preprocess Training Data ===
    preprocess_data()

    # === Step 2: Create Sequences and Train Model ===
    train_model()

    # === Step 3: Predict Unseen Data ===
    predict_unseen_data()

    # === Step 4: Evaluate and Plot ===
    evaluate_prediction_plot()



if __name__ == "__main__":
    main()
