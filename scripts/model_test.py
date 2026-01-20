import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging
import os
import json
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_model(input_path, model_path):
    try:
        df = pd.read_csv(input_path)

        X = df.drop(columns=['charges'])
        y = df['charges']

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict(X)

        metrics = {
            "MSE": mean_squared_error(y, predictions),
            "MAE": mean_absolute_error(y, predictions),
            "R2": r2_score(y, predictions)
        }

        os.makedirs("reports/plots", exist_ok=True)

        with open("reports/test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        plt.scatter(y, predictions)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Testing: Actual vs Predicted")
        plt.savefig("reports/plots/test_actual_vs_predicted.png")
        plt.close()

    except Exception as e:
        logger.error(e)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Linear Regression Model")
    parser.add_argument("--input_path", required=True, help="Path to test CSV dataset")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pkl)")
    args = parser.parse_args()
    test_model(args.input_path, args.model_path)
