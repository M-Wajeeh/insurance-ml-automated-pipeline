import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_model(input_path, model_path):
    try:
        logger.info("Loading test data from %s", input_path)
        df = pd.read_csv(input_path)

        if 'charges' not in df.columns:
            raise ValueError("Target column 'charges' not found in dataset")

        X = df.drop(columns=['charges'])
        y = df['charges']

        logger.info("Loading trained model from %s", model_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict(X)

        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        logger.info("Test MSE: %s", mse)
        logger.info("Test MAE: %s", mae)
        logger.info("Test R2: %s", r2)

    except Exception as e:
        logger.error("Model testing failed: %s", str(e))
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    test_model(args.input_path, args.model_path)
