import pandas as pd
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def feature_engineer(input_path, output_path):
    try:
        logger.info("Loading preprocessed data from %s", input_path)
        df = pd.read_csv(input_path)

        # Feature engineering
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            logger.info("Created feature: age_squared")
        else:
            logger.warning("Column 'age' not found. Skipping age_squared feature.")

        df = pd.get_dummies(df, drop_first=True)
        logger.info("Applied one-hot encoding")

        # Save engineered data
        df.to_csv(output_path, index=False)
        logger.info("Feature engineering complete. Saved to %s", output_path)

    except Exception as e:
        logger.error("Feature engineering failed: %s", str(e))
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()
    feature_engineer(args.input_path, args.output_path)
