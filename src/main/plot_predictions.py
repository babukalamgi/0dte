import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds project root to sys.path

from utils.common_imports import *

def evaluate_prediction_plot(results):

    # Plot
    for col in target_cols:
        if f"actual_{col}" in results:
            plt.figure(figsize=(6, 3))
            plt.plot(results[f"actual_{col}"], label="Actual")
            plt.plot(results[f"pred_{col}"], label="Predicted")
            plt.title(f"{col.upper()} - Actual vs Predicted")
            plt.legend()
            plt.tight_layout()
            plt.show()
