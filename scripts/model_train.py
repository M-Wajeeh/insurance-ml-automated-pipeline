import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger=logging.getLogger(__name__)

# Model Training Function
def train_model(input_path, model_path):
    try:
        # Load engineered data
        logger.info("Loading engineered data from %s", input_path)
        df = pd.read_csv(input_path)
        logger.info("Data loaded successfully with shape %s", df.shape)

        if 'charges' not in df.columns:
            raise ValueError("Target column 'charges' not found in dataset")
        # Features
        X = df.drop(columns=['charges'])  # Features
        y = df['charges']  # Target
        logger.info("features and target variable separated")

        # Define and train a linear model
        model = LinearRegression()
        model.fit(X, y)
        logger.info("model training complete")

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model training complete and saved.")
    except Exception as e:
        logger.error("Model training failed: %s", str(e))
        raise
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--input_path", help="Path to input the engineered dataset", required=True)
    parser.add_argument("--model_path", help="Path to save the trained model", required=True)
    args = parser.parse_args()
    train_model(args.input_path, args.model_path)